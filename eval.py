"""Evaluate pretrained AASIST on a random subset of ASVspoof 2019 LA eval set.

Usage:
    python eval.py --root /path/to/LA --n 500

Expected layout (standard ASVspoof 2019 LA download):
    LA/ASVspoof2019_LA_eval/flac/*.flac
    LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
Protocol line format: SPEAKER UTT_ID - ATTACK KEY   (KEY = bonafide / spoof)
"""
import argparse
import os
import random
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm

from spoof import load_model, spoof_scores


def compute_eer(labels, scores):
    """labels: 1 = bonafide, 0 = spoof. scores: higher = more bonafide."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the LA folder")
    ap.add_argument("--n", type=int, default=500, help="subset size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    proto = os.path.join(args.root, "ASVspoof2019_LA_cm_protocols",
                         "ASVspoof2019.LA.cm.eval.trl.txt")
    audio_dir = os.path.join(args.root, "ASVspoof2019_LA_eval", "flac")

    with open(proto) as f:
        rows = [l.split() for l in f if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    model, device = load_model()
    labels, scores, attacks = [], [], []
    for r in tqdm(rows):
        utt, attack, key = r[1], r[3], r[4]
        s, _ = spoof_scores(os.path.join(audio_dir, utt + ".flac"), model, device)
        labels.append(1 if key == "bonafide" else 0)
        scores.append(s)
        attacks.append(attack)

    labels, scores = np.array(labels), np.array(scores)
    print(f"\nSamples: {len(labels)} (bonafide={labels.sum()}, spoof={(1 - labels).sum()})")
    print(f"Overall EER: {compute_eer(labels, scores) * 100:.2f}%")

    # Per-attack EER (each attack vs all bonafide) - good for the report
    bona = labels == 1
    for a in sorted(set(attacks) - {"-"}):
        m = bona | (np.array(attacks) == a)
        if (np.array(attacks) == a).sum() > 5:
            print(f"  {a}: EER {compute_eer(labels[m], scores[m]) * 100:.2f}%")


if __name__ == "__main__":
    main()
