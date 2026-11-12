"""How much does the detector rely on silence?  (ASVspoof LA subset, AASIST or SSL-AASIST)

Every labeled clip is scored under four conditions, using the same code as the app (detectors.py):

  asis        the clip as it is
  trim_edges  leading + trailing silence removed (pauses inside the clip are kept)
  no_pauses   ALL quiet frames removed (speech only, what the EnergyVAD gate does)
  pad_edges   1.0 s of faint noise added to the start and the end of every clip

With --degrade, two phone-quality conditions are added (does bandwidth / codec alone flip results?):
  phone8k     clip recorded at 8 kHz (resampled 16k -> 8k -> 16k): nothing above 4 kHz
  phone_aac   8 kHz + AAC 48 kbps through ffmpeg, like a WhatsApp voice note (needs ffmpeg on PATH)
The 'narrowband' column is the share of clips that quality.py would flag as phone-quality.

"Quiet" = a 20 ms frame more than 35 dB below the clip's own 95th-percentile level.
Only the first 4 s of each processed clip is scored (the model's input size), like the live pipeline.

Reading the result:
  * If EER jumps when silence is removed or added, the detector is leaning on silence.
  * p_spoof for bonafide rising under no_pauses (and spoof falling under pad_edges)
    shows the exact direction of the shortcut.

Usage (project folder, in the venv that runs your detector):
    python silence_experiment.py --root "C:\\...\\project\\data\\LA"
    python silence_experiment.py --root "C:\\...\\project\\data\\LA" --model ssl
    python silence_experiment.py --root ... --n 200        # quicker
    python silence_experiment.py --root ... --degrade      # + phone8k / phone_aac conditions
    python silence_experiment.py --root ... --degrade --checkpoint finetune_out\\best.pth   # fine-tuned AASIST
Needs: numpy, soundfile, torch (+ what the chosen detector needs).
"""
import argparse
import csv
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from levelnorm import frame_db  # noqa: E402
import quality  # noqa: E402

SR = 16000
FR = 320
REL_QUIET_DB = 35.0
WIN = 64600                      # 4 s
CONDITIONS = ("asis", "trim_edges", "no_pauses", "pad_edges")


def quiet_mask(x):
    d = frame_db(x, FR)
    if d.size == 0:
        return d, np.zeros(0, dtype=bool)
    return d, d < (np.percentile(d, 95) - REL_QUIET_DB)


def trim_edges(x):
    d, q = quiet_mask(x)
    idx = np.where(~q)[0]
    if idx.size == 0:
        return x
    return x[idx[0] * FR:(idx[-1] + 1) * FR]


def no_pauses(x):
    d, q = quiet_mask(x)
    if d.size == 0 or q.all():
        return x
    frames = x[: d.size * FR].reshape(d.size, FR)
    return frames[~q].reshape(-1)


def pad_edges(x, rng, seconds=1.0):
    d, _ = quiet_mask(x)
    floor_db = float(np.clip(np.percentile(d, 10), -70.0, -45.0)) if d.size else -60.0
    n = int(seconds * SR)
    noise = lambda: (10 ** (floor_db / 20) * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
    return np.concatenate([noise(), x, noise()])


def phone8k(x):
    from scipy.signal import resample_poly
    return resample_poly(resample_poly(x, 1, 2), 2, 1).astype(np.float32)


def phone_aac(x, bitrate="48k"):
    """8 kHz + AAC through ffmpeg and back to 16 kHz. None if ffmpeg fails."""
    import shutil
    import subprocess
    import tempfile
    import soundfile as sf
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    with tempfile.TemporaryDirectory() as d:
        a, b, c = (os.path.join(d, n) for n in ("in.wav", "mid.m4a", "out.wav"))
        sf.write(a, x, SR)
        r1 = subprocess.run([ff, "-y", "-v", "error", "-i", a, "-ar", "8000", "-ac", "1",
                             "-c:a", "aac", "-b:a", bitrate, b], capture_output=True)
        r2 = subprocess.run([ff, "-y", "-v", "error", "-i", b, "-ar", "16000", "-ac", "1", c],
                            capture_output=True)
        if r1.returncode or r2.returncode or not os.path.isfile(c):
            return None
        y, _ = sf.read(c, dtype="float32")
        return y


def transform(name, x, rng):
    if name == "asis":
        return x
    if name == "trim_edges":
        return trim_edges(x)
    if name == "no_pauses":
        return no_pauses(x)
    if name == "phone8k":
        return phone8k(x)
    if name == "phone_aac":
        return phone_aac(x)
    return pad_edges(x, rng)


def compute_eer(labels, scores):
    """labels: 1 = bonafide, 0 = spoof; higher score = more bonafide."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    best, eer = 1e9, float("nan")
    for t in np.unique(scores):
        frr = float(np.mean(pos < t))           # bonafide rejected
        far = float(np.mean(neg >= t))          # spoof accepted
        if abs(frr - far) < best:
            best, eer = abs(frr - far), (frr + far) / 2
    return eer


def load_labels(root, protocols):
    protos = list(protocols) or sorted(
        glob.glob(os.path.join(root, "**", "ASVspoof2019.LA.cm.*.txt"), recursive=True))
    labels = {}
    for p in protos:
        for line in open(p):
            f = line.split()
            if len(f) >= 5 and f[4] in ("bonafide", "spoof"):
                labels[f[1]] = f[4]
        print("protocol:", p)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LA folder (contains the flac folder + protocols)")
    ap.add_argument("--protocol", action="append", default=[])
    ap.add_argument("--model", choices=["aasist", "ssl"], default="aasist")
    ap.add_argument("--n", type=int, default=0, help="max clips (0 = all labeled clips found)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--degrade", action="store_true", help="also score phone8k and phone_aac versions")
    ap.add_argument("--checkpoint", help="fine-tuned AASIST weights (.pth); only with --model aasist")
    ap.add_argument("--csv", default="silence_experiment.csv")
    args = ap.parse_args()

    import soundfile as sf
    import detectors

    labels = load_labels(args.root, args.protocol)
    files = sorted(glob.glob(os.path.join(args.root, "**", "*.flac"), recursive=True))
    items = [(f, labels[os.path.splitext(os.path.basename(f))[0]]) for f in files
             if os.path.splitext(os.path.basename(f))[0] in labels]
    if not items:
        sys.exit("No labeled flac clips found: check --root / --protocol.")
    random.Random(args.seed).shuffle(items)
    if args.n:
        items = items[: args.n]
    nb = sum(1 for _, k in items if k == "bonafide")
    print(f"{len(items)} clips (bonafide={nb}, spoof={len(items) - nb}), model={args.model}")
    if nb < 30:
        print("Warning: few bonafide clips: EER will be noisy.")

    conditions = list(CONDITIONS)
    if args.degrade:
        import shutil
        conditions.append("phone8k")
        if shutil.which("ffmpeg"):
            conditions.append("phone_aac")
        else:
            print("ffmpeg not found on PATH: skipping phone_aac (PowerShell: $env:PATH += \";$PWD\\bin\")")

    if args.checkpoint:
        if args.model != "aasist":
            sys.exit("--checkpoint is only for --model aasist")
        import aasist_io
        model, device = aasist_io.load(args.checkpoint)
    else:
        model, device = detectors.load_model(args.model)
    rng = np.random.default_rng(args.seed)
    rows = []
    for i, (f, key) in enumerate(items):
        x, sr = sf.read(f, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            print("skip (sample rate)", f)
            continue
        row = {"utt": os.path.splitext(os.path.basename(f))[0], "key": key}
        for c in conditions:
            y = transform(c, x, rng)
            if y is None or y.size < SR // 4:       # nothing left / ffmpeg failed: use the original
                y = x
            row[c] = detectors.spoof_prob_window(model, device, y[:WIN])
            row[c + "_nb"] = int(quality.check(y[:WIN])["narrowband"])
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(items)}", flush=True)

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["utt", "key", *[k for c in conditions for k in (c, c + "_nb")]])
        w.writeheader()
        w.writerows(rows)

    y = np.array([1 if r["key"] == "bonafide" else 0 for r in rows])
    print(f"\n{'condition':12s} {'EER %':>7s} {'bonafide p_spoof':>18s} {'spoof p_spoof':>15s}"
          f" {'bonafide flagged':>18s} {'spoof caught':>14s} {'narrowband':>11s}")
    for c in conditions:
        p = np.array([r[c] for r in rows])
        eer = compute_eer(y, 1.0 - p) * 100
        b, s = p[y == 1], p[y == 0]
        print(f"{c:12s} {eer:7.1f} {np.median(b):18.2f} {np.median(s):15.2f}"
              f" {100 * np.mean(b > 0.5):17.0f}% {100 * np.mean(s > 0.5):13.0f}%"
              f" {100 * np.mean([r[c + '_nb'] for r in rows]):10.0f}%")
    print(f"\n(p_spoof columns are medians; 'flagged'/'caught' use a 0.5 threshold.)"
          f"\nPer-clip scores saved to {args.csv}")


if __name__ == "__main__":
    main()