"""Level / pause statistics of an ASVspoof 2019 LA subset, to calibrate LevelNormalizer.target_db.

Usage (from the project folder, PowerShell):
    python calibrate_level.py --root ..\\data\\LA
    python calibrate_level.py --root ..\\data\\LA --rec rec.aac

Layout it expects (same as make_subset.py / eval.py):
    ROOT/ASVspoof2019_LA_eval/flac/*.flac
    ROOT/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt   (SPEAKER UTT - ATTACK KEY)
If the protocol file is missing, every flac in --audio-dir/ROOT is treated as 'unknown'.

What it prints, per class (bonafide / spoof):
  p95_db      95th-percentile 20 ms frame level  -> the number LevelNormalizer aims at (target_db)
  quiet%      frames more than 35 dB under the clip's own p95 (pauses / silence)
  edge_s      leading + trailing silence in seconds (a known ASVspoof shortcut)
  dur_s       clip length
and, with --rec, where your recording falls inside the bonafide distribution.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from levelnorm import frame_db  # noqa: E402

SR = 16000
FR = 320          # 20 ms
REL_QUIET_DB = 35.0


def clip_stats(x):
    d = frame_db(np.asarray(x, dtype=np.float32), FR)
    if d.size == 0:
        return None
    p95 = float(np.percentile(d, 95))
    quiet = d < (p95 - REL_QUIET_DB)
    loud_idx = np.where(~quiet)[0]
    if loud_idx.size:
        edge_frames = loud_idx[0] + (d.size - 1 - loud_idx[-1])
    else:
        edge_frames = d.size
    return {"p95_db": p95, "quiet_pct": 100.0 * float(quiet.mean()),
            "edge_s": edge_frames * FR / SR, "dur_s": len(x) / SR}


def load_flac(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    return x


def summarize(name, rows):
    if not rows:
        return
    print(f"\n{name}  (n={len(rows)})")
    print(f"  {'':10s} {'p10':>8s} {'median':>8s} {'p90':>8s}")
    for key, label in (("p95_db", "p95_db"), ("quiet_pct", "quiet%"), ("edge_s", "edge_s"), ("dur_s", "dur_s")):
        v = np.array([r[key] for r in rows])
        print(f"  {label:10s} {np.percentile(v, 10):8.1f} {np.median(v):8.1f} {np.percentile(v, 90):8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LA folder (see layout above)")
    ap.add_argument("--audio-dir", default=None, help="override the flac folder")
    ap.add_argument("--protocol", action="append", default=[],
                    help="protocol .txt with 'SPEAKER UTT - ATTACK KEY' lines (repeatable). "
                         "Default: every ASVspoof2019.LA.cm.*.txt found under --root")
    ap.add_argument("--rec", default=None, help="your own recording to compare (aac/wav/...)")
    ap.add_argument("--csv", default="level_stats.csv")
    args = ap.parse_args()

    audio_dir = args.audio_dir or os.path.join(args.root, "ASVspoof2019_LA_eval", "flac")
    if not os.path.isdir(audio_dir):
        audio_dir = args.root
    protos = list(args.protocol) or sorted(
        glob.glob(os.path.join(args.root, "**", "ASVspoof2019.LA.cm.*.txt"), recursive=True))
    labels = {}
    for proto in protos:
        for line in open(proto):
            p = line.split()
            if len(p) >= 5 and p[4] in ("bonafide", "spoof"):
                labels[p[1]] = p[4]            # utt id -> bonafide / spoof
        print("protocol:", proto)
    if not labels:
        print("No protocol labels found: all clips treated as 'unknown'. Pass --protocol FILE.txt")

    files = sorted(glob.glob(os.path.join(audio_dir, "**", "*.flac"), recursive=True))
    if not files:
        sys.exit(f"No .flac files under {audio_dir}")
    print(f"{len(files)} flac files, {len(labels)} labels")

    rows = []
    for i, f in enumerate(files):
        utt = os.path.splitext(os.path.basename(f))[0]
        try:
            s = clip_stats(load_flac(f))
        except Exception as e:                      # noqa: BLE001
            print("skip", f, e)
            continue
        if s:
            s.update(utt=utt, key=labels.get(utt, "unknown"))
            rows.append(s)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(files)}", flush=True)

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["utt", "key", "p95_db", "quiet_pct", "edge_s", "dur_s"])
        w.writeheader()
        w.writerows(rows)

    n_lab = sum(1 for r in rows if r["key"] != "unknown")
    print(f"{n_lab} of {len(rows)} clips matched a protocol label")
    bona = [r for r in rows if r["key"] == "bonafide"]
    spoof = [r for r in rows if r["key"] == "spoof"]
    unk = [r for r in rows if r["key"] == "unknown"]
    summarize("BONAFIDE", bona)
    summarize("SPOOF", spoof)
    summarize("UNKNOWN", unk)

    ref = bona or unk
    if bona:
        t = float(np.median([r["p95_db"] for r in bona]))
        print(f"\nSuggested LevelNormalizer target_db (median bonafide p95_db): {t:.1f}")
        if spoof:
            ts = float(np.median([r["p95_db"] for r in spoof]))
            print(f"Median spoof p95_db: {ts:.1f}  (difference {ts - t:+.1f} dB: a gap here means level alone separates the classes)")
        eb = np.median([r["edge_s"] for r in bona])
        if spoof:
            es = np.median([r["edge_s"] for r in spoof])
            print(f"Median leading+trailing silence: bonafide {eb:.2f}s vs spoof {es:.2f}s")

    if args.rec and ref:
        try:
            from capture import load_audio_16k      # handles aac/m4a via ffmpeg
            wav = load_audio_16k(args.rec)
        except ImportError:                          # librosa missing: fall back to plain wav/flac
            wav = load_flac(args.rec)
        s = clip_stats(wav)
        v = np.array([r["p95_db"] for r in ref])
        pct = 100.0 * float((v < s["p95_db"]).mean())
        print(f"\nYOUR RECORDING {os.path.basename(args.rec)}: p95_db {s['p95_db']:.1f}  quiet% {s['quiet_pct']:.0f}  "
              f"edge_s {s['edge_s']:.2f}  dur {s['dur_s']:.1f}s")
        print(f"  its level is above {pct:.0f}% of the {'bonafide' if bona else 'all'} clips "
              f"(median {np.median(v):.1f} dB).")
    print(f"\nPer-clip numbers saved to {args.csv}")


if __name__ == "__main__":
    main()