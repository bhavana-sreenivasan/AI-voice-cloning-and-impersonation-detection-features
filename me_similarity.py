"""Which parts of a recording sound like YOU?  Picks the --drop-above threshold from data.

live_cli.py --me / --me-file throws away every 1 s slice whose cosine similarity to your enrolled voice
is >= --drop-above (default = speaker.MATCH_THRESHOLD, which is UNCALIBRATED for ECAPA2).  This prints
that similarity for every 1 s slice of the recording, using the same embedding code, so you can see
where your voice sits compared with the other person's and choose a threshold that separates them.

Usage (project folder, in the venv that has ECAPA2 / SpeechBrain):
    python me_similarity.py rec.aac --me-file pages\\user.mp3
    python me_similarity.py rec.aac --me-file pages\\user.mp3 --slice 1.5

Then e.g.:  python live_cli.py --file rec.aac --me-file pages\\user.mp3 --drop-above 0.45 --fast
"""
import argparse

import numpy as np

SR = 16000
FR = 320


def level_db(x):
    n = len(x) // FR
    if n == 0:
        return -120.0
    f = x[: n * FR].reshape(n, FR)
    return float(np.percentile(10 * np.log10(np.mean(f * f, axis=1) + 1e-10), 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="rec.aac")
    ap.add_argument("--me-file", required=True, help="a clean recording of YOUR voice")
    ap.add_argument("--slice", type=float, default=1.0, help="slice length in seconds (live_cli uses 1.0)")
    ap.add_argument("--silence-db", type=float, default=-50.0, help="slices quieter than this are skipped")
    args = ap.parse_args()

    import torch
    import speaker
    from capture import load_audio_16k

    emb = lambda w: speaker.embed(torch.tensor(np.asarray(w, dtype=np.float32))).numpy()   # noqa: E731
    me = load_audio_16k(args.me_file)[: 30 * SR]
    ref = emb(me)
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    cos = lambda e: float(ref @ (e / (np.linalg.norm(e) + 1e-12)))                          # noqa: E731

    n = int(args.slice * SR)
    own = [cos(emb(me[i:i + n])) for i in range(0, len(me) - n + 1, n)]
    if own:
        print(f"Your reference file against itself (1 slice at a time, optimistic): "
              f"median {np.median(own):.2f}, min {np.min(own):.2f}\n")

    x = np.asarray(load_audio_16k(args.file), dtype=np.float32)
    print(f"{args.file}: {len(x) / SR:.1f}s   backend={speaker.BACKEND}   "
          f"default drop threshold={speaker.MATCH_THRESHOLD:.2f}\n")
    print(f"{'time':>7} {'level':>7} {'sim':>6}  ")
    sims = []
    for i in range(0, len(x) - n + 1, n):
        piece = x[i:i + n]
        lv = level_db(piece)
        if lv < args.silence_db:
            print(f"{i / SR:6.1f}s {lv:6.1f}  (silence)")
            continue
        s = cos(emb(piece))
        sims.append(s)
        print(f"{i / SR:6.1f}s {lv:6.1f} {s:+6.2f}  {'#' * int(max(0.0, s) * 40)}")

    if not sims:
        print("\nNo speech slices found.")
        return
    v = np.sort(np.array(sims))
    print(f"\n{len(v)} speech slices.  similarity percentiles: "
          + "  ".join(f"p{q}={np.percentile(v, q):+.2f}" for q in (10, 25, 50, 75, 90)))
    print("share of slices dropped at each --drop-above:")
    for t in (0.15, 0.25, 0.35, 0.45, 0.55):
        print(f"   {t:.2f}: {100 * np.mean(v >= t):4.0f}% dropped")
    gaps = np.diff(v)
    if len(v) >= 4:
        k = int(np.argmax(gaps))
        print(f"\nLargest gap between neighbouring similarities: {v[k]:+.2f} -> {v[k + 1]:+.2f}  "
              f"(a threshold near {(v[k] + v[k + 1]) / 2:+.2f} would split the slices into two groups).")
        print("Only trust that if the slices above it really are YOUR turns (listen to the file at those times).")
    print(f"\nIf almost everything is dropped at {speaker.MATCH_THRESHOLD:.2f}, that is why live_cli printed no scores.")

def slice_label(segs, user, t0, t1):
    """segs: [(start_s, end_s, speaker)]. 'user' / 'opponent' if exactly one speaker covers the whole slice, else None."""
    who = set()
    for s, e, spk in segs:
        if s < t1 and e > t0:
            if s > t0 or e < t1:        # the turn starts or ends inside the slice: not a clean slice
                return None
            who.add(spk == user)
    if len(who) != 1:
        return None
    return "user" if next(iter(who)) else "opponent"


def auc(pos, neg):
    """Probability that a random 'pos' score is higher than a random 'neg' score (0.5 = no separation)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

if __name__ == "__main__":
    main()