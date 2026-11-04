"""Calibrate the speaker-match thresholds for the active backend (ECAPA2 by default).

Layout:  <root>/<speaker_name>/*.wav   (>= 5 speakers, >= 3 clips each, e.g. a slice of VoxCeleb1
or LibriSpeech; ideally ALSO a phone-codec copy of the same clips, see below).

    python calibrate_speaker.py --root data/spk
    SPEAKER_BACKEND=speechbrain python calibrate_speaker.py --root data/spk   # compare backends

Prints the EER and the values to export as SPEAKER_MATCH_T / SPEAKER_MARGIN.
Thresholds measured on clean audio will be too optimistic for phone calls: re-run on 8 kHz
G.711/AMR/Opus-processed copies of the clips before trusting them for live-call decisions.
"""
import argparse
import itertools
import os

import numpy as np


def suggest_thresholds(same, diff, far=0.01, frr=0.01):
    """same / diff: cosine scores of same-speaker and different-speaker pairs.

    'match' above t_strict (only `far` of impostors reach it); 'mismatch' below t_loose (only `frr`
    of genuine pairs fall under it); in between = 'uncertain'. Returns dict with eer, match_t, margin.
    """
    same, diff = np.asarray(same, float), np.asarray(diff, float)
    cand = np.unique(np.concatenate([same, diff]))
    fa = np.array([(diff >= t).mean() for t in cand])
    fr = np.array([(same < t).mean() for t in cand])
    i = int(np.argmin(np.abs(fa - fr)))
    eer, t_eer = float((fa[i] + fr[i]) / 2), float(cand[i])
    t_strict = float(np.quantile(diff, 1 - far))
    t_loose = float(np.quantile(same, frr))
    if t_strict > t_loose:
        match_t, margin = (t_strict + t_loose) / 2, (t_strict - t_loose) / 2
    else:                       # classes separate cleanly: no real uncertain zone needed
        match_t, margin = t_eer, 0.02
    return {"eer": eer, "eer_threshold": t_eer, "match_t": match_t, "margin": margin,
            "t_strict": t_strict, "t_loose": t_loose}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    import speaker
    embs = {}
    for spk in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, spk)
        if os.path.isdir(d):
            embs[spk] = [speaker.embed(speaker.load_audio(os.path.join(d, f))).numpy()
                         for f in sorted(os.listdir(d))]
    same = [float(a @ b) for v in embs.values() for a, b in itertools.combinations(v, 2)]
    diff = [float(a @ b) for (_, va), (_, vb) in itertools.combinations(embs.items(), 2)
            for a in va for b in vb]
    r = suggest_thresholds(same, diff)
    print(f"backend={speaker.BACKEND}  speakers={len(embs)}  same-pairs={len(same)}  diff-pairs={len(diff)}")
    print(f"same-speaker mean={np.mean(same):.3f}  different-speaker mean={np.mean(diff):.3f}")
    print(f"EER={r['eer'] * 100:.2f}% at cosine {r['eer_threshold']:.3f}")
    print(f"\nexport SPEAKER_MATCH_T={r['match_t']:.3f}\nexport SPEAKER_MARGIN={r['margin']:.3f}")


if __name__ == "__main__":
    main()
