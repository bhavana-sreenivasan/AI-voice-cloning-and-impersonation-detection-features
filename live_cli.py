"""Live monitor in the terminal (no Streamlit needed).

  # Test on a recording, paced like a real call; remote party is channel 0 of the stereo file:
  python live_cli.py --file call.wav --channel 0 --detector aasist

  # One mixed recording (you + the caller): remove YOUR voice using a clean sample of it:
  python live_cli.py --file call.wav --me-file my_voice.wav --normalize

  # Live: analyse what the computer is playing (= the remote party only):
  python live_cli.py --loopback --detector ssl

  --no-speaker   skip the session voice-swap tracker       --no-vad   score everything, incl. silence
  --normalize    scale speech to a steady level before scoring (see levelnorm.py)
"""
import argparse
import sys

import numpy as np

import detectors
from levelnorm import LevelNormalizer
from live import EnergyVAD, GateChain, LiveAnalyzer, PauseLimiter, SessionVoiceTracker, UserVoiceFilter
from live_session import LiveSession

ICON = {"Low": "OK  ", "Uncertain": "??  ", "High": "ALRT"}


def build(args):
    model, device = detectors.load_model(args.detector)
    score_fn = lambda w: detectors.spoof_prob_window(model, device, w)    # noqa: E731
    tracker = None
    if args.claim or args.claim_file:     # identity check: is the caller who they claim to be?
        import speaker
        import voice_store
        if args.claim_file:
            import torch
            from capture import load_audio_16k
            ref = speaker.embed(torch.tensor(load_audio_16k(args.claim_file)[: 30 * 16000]))
        else:
            paths = voice_store.clips(args.claim)
            if not paths:
                sys.exit(f"No saved voice named '{args.claim}' in the voice library.")
            ref = speaker.enroll(paths)
        tracker = lambda w: speaker.cosine(speaker.embed(w), ref)         # noqa: E731
        print("Comparing the caller with the claimed person's saved voice.", flush=True)
    elif not args.no_speaker:
        import speaker
        embed_fn = lambda w: speaker.embed(w).numpy()                     # noqa: E731
        tracker = SessionVoiceTracker(embed_fn, change_below=speaker.MATCH_THRESHOLD - speaker.UNCERTAIN_MARGIN)
    removing_user = bool(args.me_file or args.me)
    if args.no_vad or args.gate == "none":
        gates = []
    elif args.gate == "pause" and not removing_user:
        gates = [PauseLimiter()]
    else:                                  # speech-only is needed to tell speakers apart (--me)
        gates = [EnergyVAD()]
        if args.gate == "pause":
            print("Note: with --me the audio is speech-only (pauses are removed).", flush=True)
    if args.me_file or args.me:                  # remove the local user's own voice
        import speaker
        import torch
        import voice_store
        if args.me_file:
            from capture import load_audio_16k
            wav = load_audio_16k(args.me_file)[: 30 * 16000]      # up to 30 s is plenty
            ref = speaker.embed(torch.tensor(wav)).numpy()
        else:
            ref = speaker.enroll(voice_store.clips(args.me)).numpy()
        t = speaker.MATCH_THRESHOLD if args.drop_above is None else args.drop_above
        gates.append(UserVoiceFilter(lambda w: speaker.embed(w).numpy(), ref, drop_above=t))
        print(f"Ignoring the local user's voice (similarity >= {t:.2f} is dropped).", flush=True)
    if args.normalize:                           # steady speech level before the spoof model
        gates.append(LevelNormalizer(target_db=args.target_db))
        print(f"Level normalisation on (target {args.target_db:.1f} dB).", flush=True)
    return LiveAnalyzer(score_fn, tracker,
        gate=GateChain(*gates) if gates else None,
        smooth=3, check_quality=True)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--loopback", action="store_true")
    ap.add_argument("--channel", default="mix", help="0, 1 or mix (for --file)")
    ap.add_argument("--detector", choices=["aasist", "ssl"], default=detectors.default_name())
    ap.add_argument("--hop", type=float, default=1.0, help="seconds between scores (raise on slow CPUs)")
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--me", help="name of YOUR voice in the saved-voice library: it is removed from the stream")
    ap.add_argument("--me-file", help="a clean recording of YOUR voice: it is removed from the stream")
    ap.add_argument("--drop-above", type=float, default=None,
                    help="cosine similarity to your voice above which audio is dropped (calibrate it)")
    ap.add_argument("--claim", help="saved-voice name of the person the caller CLAIMS to be (e.g. your son)")
    ap.add_argument("--claim-file", help="a clean recording of the person the caller claims to be")
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--gate", choices=["pause", "vad", "none"], default="pause",
                    help="pause: keep natural pauses, cap long silences (default); vad: speech only; none: score everything")
    ap.add_argument("--no-vad", action="store_true", help="same as --gate none")
    ap.add_argument("--normalize", action="store_true",
                    help="scale speech to a steady level before scoring (reduces loudness sensitivity)")
    ap.add_argument("--target-db", type=float, default=-20.0,
                    help="speech level to scale to (95th-percentile frame level, dB); see levelnorm.py")
    ap.add_argument("--fast", action="store_true", help="--file only: don't pace in real time")
    args = ap.parse_args()

    analyzer = build(args)
    claimed = bool(args.claim or args.claim_file)
    if args.file:
        from capture import ArraySource
        source = ArraySource.from_file(args.file, args.channel, realtime=not args.fast)
    else:
        from capture import LoopbackSource
        source = LoopbackSource()

    def show(e):
        sim = "  -  " if e["similarity"] is None else f"{e['similarity']:+.2f}"
        if claimed and e["similarity"] is not None:
            import speaker
            sim += f" [{speaker.decide(e['similarity'])}]"
        print(f"[{e['t']:6.1f}s] {ICON[e['risk']]} spoof={e['p_spoof']:.2f} voice-sim={sim} "
              f"speech={e['speech_s']:.0f}s lat={e['latency_ms']:.0f}ms"
              + (f" user-removed={e['user_dropped_s']:.0f}s" if e["user_dropped_s"] else "")
              + ("  << NEW ALERT" if e["new_alert"] else "")
              + ("  << SPEAKER CHANGED" if e["speaker_change"] else ""), flush=True)

    sess = LiveSession(analyzer, source, on_event=show, drop_oldest=not args.fast).start()
    try:
        while sess.running:
            sess.join(0.5)
    except KeyboardInterrupt:
        sess.stop()
    s = analyzer.summary()
    print("\nSummary:", s if s else "no speech analysed", f"| dropped chunks: {sess.dropped_chunks}")
    if claimed:
        import speaker
        sims = [e["similarity"] for e in analyzer.history if e["similarity"] is not None]
        if sims:
            m = sum(sims) / len(sims)
            print(f"Identity vs claimed voice: mean similarity {m:.2f} -> {speaker.decide(m)}")
    if sess.error:
        raise sess.error


if __name__ == "__main__":
    main()