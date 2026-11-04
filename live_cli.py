"""Live monitor in the terminal (no Streamlit needed).

  # Test on a recording, paced like a real call; remote party is channel 0 of the stereo file:
  python live_cli.py --file call.wav --channel 0 --detector aasist

  # Live: analyse what the computer is playing (= the remote party only):
  python live_cli.py --loopback --detector ssl

  --no-speaker   skip the session voice-swap tracker       --no-vad   score everything, incl. silence
"""
import argparse

import numpy as np

import detectors
from live import EnergyVAD, LiveAnalyzer, SessionVoiceTracker
from live_session import LiveSession

ICON = {"Low": "OK  ", "Uncertain": "??  ", "High": "ALRT"}


def build(args):
    model, device = detectors.load_model(args.detector)
    score_fn = lambda w: detectors.spoof_prob_window(model, device, w)    # noqa: E731
    tracker = None
    if not args.no_speaker:
        import speaker
        embed_fn = lambda w: speaker.embed(w).numpy()                     # noqa: E731
        tracker = SessionVoiceTracker(embed_fn, change_below=speaker.MATCH_THRESHOLD - speaker.UNCERTAIN_MARGIN)
    return LiveAnalyzer(score_fn, tracker, gate=None if args.no_vad else EnergyVAD(),
                        hop_s=args.hop, smooth=args.smooth)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file")
    src.add_argument("--loopback", action="store_true")
    ap.add_argument("--channel", default="mix", help="0, 1 or mix (for --file)")
    ap.add_argument("--detector", choices=["aasist", "ssl"], default=detectors.default_name())
    ap.add_argument("--hop", type=float, default=1.0, help="seconds between scores (raise on slow CPUs)")
    ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--no-speaker", action="store_true")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--fast", action="store_true", help="--file only: don't pace in real time")
    args = ap.parse_args()

    analyzer = build(args)
    if args.file:
        from capture import ArraySource
        source = ArraySource.from_file(args.file, args.channel, realtime=not args.fast)
    else:
        from capture import LoopbackSource
        source = LoopbackSource()

    def show(e):
        sim = "  -  " if e["similarity"] is None else f"{e['similarity']:+.2f}"
        print(f"[{e['t']:6.1f}s] {ICON[e['risk']]} spoof={e['p_spoof']:.2f} voice-sim={sim} "
              f"speech={e['speech_s']:.0f}s lat={e['latency_ms']:.0f}ms"
              + ("  << NEW ALERT" if e["new_alert"] else "")
              + ("  << SPEAKER CHANGED" if e["speaker_change"] else ""), flush=True)

    sess = LiveSession(analyzer, source, on_event=show).start()
    try:
        while sess.running:
            sess.join(0.5)
    except KeyboardInterrupt:
        sess.stop()
    s = analyzer.summary()
    print("\nSummary:", s if s else "no speech analysed", f"| dropped chunks: {sess.dropped_chunks}")
    if sess.error:
        raise sess.error


if __name__ == "__main__":
    main()
