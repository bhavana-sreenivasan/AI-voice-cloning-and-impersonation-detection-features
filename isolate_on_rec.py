"""Does isolating ONE speaker of a two-person recording change the spoof score?  (one file, 4 s windows)

The recording is split into two voices with the project's own method (diarize.py: speaker embeddings +
clustering).  Then it is scored in four ways:

  mix         the recording as it is (both voices)
  A_concat    speaker A's speech glued together, pauses removed   <- what the app's phone-call mode scores
  B_concat    the same for speaker B
  A_timeline  original timing kept; the OTHER speaker's turns replaced by faint room noise
  B_timeline  the same for speaker B

Reading it:
  * *_concat close to 1.00 even for a real voice -> removing pauses is what breaks the detectors
    (the same effect as the no_pauses test), so isolation as implemented makes things worse.
  * *_timeline keeps each speaker's own pauses but also turns the other speaker's turns into silence,
    so it shows what the detector says about one voice with natural gaps.
  * If one speaker is scored fake and the other real in the timeline variants, the detector reacts
    to that voice's recording path (e.g. phone codec) and not to the content.

Usage (project folder, in the venv with ECAPA2 / SpeechBrain + librosa):
    python isolate_on_rec.py rec.aac --model aasist
    python isolate_on_rec.py rec.aac --model ssl
"""
import argparse
import sys

import numpy as np

SR = 16000
WIN = 64600           # 4 s
FR = 320


def merge_close(spans, gap):
    """Join speech spans separated by less than `gap` samples (choppy / noise-gated phone audio)."""
    out = [list(spans[0])]
    for st, en in spans[1:]:
        if st - out[-1][1] <= gap:
            out[-1][1] = en
        else:
            out.append([st, en])
    return out


def speaker_mask(wav, merge_gap_s=0.5, verbose=True):
    """Per-sample speaker label (0/1, -1 = pause / not assigned) with diarize.py's own method,
    plus merging of speech spans closer than `merge_gap_s` (diarize.py does not merge them, so
    choppy audio leaves too few 1.5 s windows to cluster).
    Returns (mask, cosine_between_speakers); (None, None) when it cannot split the audio."""
    import librosa
    import torch
    from sklearn.cluster import AgglomerativeClustering
    import diarize
    import speaker

    spans = librosa.effects.split(wav, top_db=30)
    n_raw = len(spans)
    if merge_gap_s > 0 and n_raw:
        spans = merge_close(spans, int(merge_gap_s * SR))
    wins = [(st, st + diarize.WIN) for s, e in spans for st in range(s, e - diarize.WIN + 1, diarize.HOP)]
    if verbose:
        print(f"speech spans: {n_raw} -> {len(spans)} after merging gaps < {merge_gap_s:.1f}s; "
              f"{len(wins)} windows of {diarize.WIN / SR:.1f}s to cluster (need at least 4)")
    if len(wins) < 4:
        return None, None
    embs = np.stack([speaker.embed(torch.tensor(wav[a:b])).numpy() for a, b in wins])
    labels = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(embs)
    if min((labels == 0).sum(), (labels == 1).sum()) < 2:
        return None, None
    c = [embs[labels == k].mean(0) for k in (0, 1)]
    c = [v / np.linalg.norm(v) for v in c]
    cos = float(c[0] @ c[1])
    mask = np.full(len(wav), -1)
    for (a, b), k in zip(wins, labels):
        mid = (a + b) // 2
        mask[mid - diarize.HOP // 2: mid + diarize.HOP // 2] = k
    return mask, cos


def diarize_same_cos():
    import diarize
    return float(diarize.SAME_SPEAKER_COS)


def room_noise(x, n, rng):
    from levelnorm import frame_db
    d = frame_db(x, FR)
    floor_db = float(np.clip(np.percentile(d, 10), -70.0, -45.0)) if d.size else -60.0
    return (10 ** (floor_db / 20) * rng.standard_normal(n)).astype(np.float32)


def timeline_only(x, mask, k, rng):
    """Keep the original timing; the other speaker's samples become faint room noise."""
    y = x.copy()
    other = mask == (1 - k)
    y[other] = room_noise(x, int(other.sum()), rng)
    return y


def window_scores(detectors, model, device, y):
    y = np.asarray(y, dtype=np.float32)
    starts = [0] if y.size <= WIN else list(range(0, y.size - WIN + 1, SR))
    return starts, np.array([detectors.spoof_prob_window(model, device, y[s:s + WIN]) for s in starts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="rec.aac")
    ap.add_argument("--model", choices=["aasist", "ssl"], default="aasist")
    ap.add_argument("--merge-gap", type=float, default=0.5,
                    help="join speech spans closer than this many seconds before clustering (0 = off)")
    args = ap.parse_args()

    import detectors
    from capture import load_audio_16k

    x = np.asarray(load_audio_16k(args.file), dtype=np.float32)
    print(f"{args.file}: {x.size / SR:.1f}s, model={args.model}")
    mask, cos = speaker_mask(x, args.merge_gap)
    if mask is None:
        sys.exit("Could not split the audio: too few windows, or one cluster is almost empty "
                 "(only one speaker, or too little speech). Try --merge-gap 1.0.")
    share = [float(np.mean(mask == k)) * x.size / SR for k in (0, 1)]
    print(f"speaker A: {share[0]:.1f}s of speech   speaker B: {share[1]:.1f}s   "
          f"(similarity between the two voices {cos:.2f}; diarize.py treats >{diarize_same_cos():.2f} as one speaker)")
    if cos > diarize_same_cos():
        print("WARNING: the two clusters are about as similar as one voice to itself, so this may be a single "
              "speaker split at random: do not read A and B as two people.")
    print()

    rng = np.random.default_rng(42)
    model, device = detectors.load_model(args.model)
    variants = {
        "mix": x,
        "A_concat": x[mask == 0], "B_concat": x[mask == 1],
        "A_timeline": timeline_only(x, mask, 0, rng), "B_timeline": timeline_only(x, mask, 1, rng),
    }
    scores = {}
    print(f"{'variant':<12}{'dur_s':>7}{'windows':>9}{'mean':>7}{'median':>8}{'max':>7}{'>0.5':>7}")
    for name, y in variants.items():
        if y.size < SR:
            print(f"{name:<12}{y.size / SR:>7.1f}   too short to score")
            continue
        starts, s = window_scores(detectors, model, device, y)
        scores[name] = (starts, s)
        print(f"{name:<12}{y.size / SR:>7.1f}{s.size:>9}{s.mean():>7.2f}{np.median(s):>8.2f}{s.max():>7.2f}{(s > 0.5).mean():>7.0%}")

    if "mix" in scores:
        print("\nper window (original timeline):  who talks | spoof score of each variant")
        print(f"{'start':>6} {'A%':>5} {'B%':>5} {'mix':>6} {'A_only':>7} {'B_only':>7}")
        starts, mix = scores["mix"]
        for i, st in enumerate(starts):
            seg = mask[st:st + WIN]
            row = [f"{st / SR:5.1f}s", f"{100 * np.mean(seg == 0):4.0f}%", f"{100 * np.mean(seg == 1):4.0f}%", f"{mix[i]:6.2f}"]
            for v in ("A_timeline", "B_timeline"):
                row.append(f"{scores[v][1][i]:7.2f}" if v in scores and i < len(scores[v][1]) else f"{'-':>7}")
            print(" ".join(row))


if __name__ == "__main__":
    main()