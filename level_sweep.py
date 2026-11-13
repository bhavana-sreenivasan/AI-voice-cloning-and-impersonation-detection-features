"""level_sweep.py - is there a loudness at which AASIST is right on real calls AND still catches fakes?

eval_callhome.py showed that on 5 real calls the share of windows scored fake rises with the call's level
(r = +0.86).  This scales every clip to a chosen level before scoring, with the SAME gain for a whole
call / clip (the best case for any normaliser: it knows the level perfectly), so you can see:

  CallHome (all real)   share of windows scored fake: mean over calls, and the worst call   -> want LOW
  LA real clips         share scored fake                                                  -> want LOW
  LA fake clips         share scored fake ("caught")                                       -> want HIGH
  LA EER                the benchmark error at that level

Level = 95th-percentile 20 ms frame level in dB (p95_db in levelnorm.py).  Natural levels: ASVspoof clips
about -12 dB, your CallHome calls -22 to -32 dB, your WhatsApp clip -21 dB.
A 'natural' row (no gain change) is always included.

It needs only AASIST inference.  Rough cost: (targets + 1) x (calls x windows + LA clips) windows;
the defaults are about 1200 windows (roughly 10 minutes on a 2-thread CPU).  Use --quick for a fast look.

    python level_sweep.py --calls-dir callhome_out --root "C:\\...\\project\\data\\LA"
    python level_sweep.py --calls-dir callhome_out --root "C:\\...\\project\\data" --protocol "C:\\...\\ASVspoof2019.LA.cm.eval.trl.txt"   # more real LA clips
    python level_sweep.py --calls-dir callhome_out --root "C:\\...\\data\\LA" --la-phone     # LA clips as 8 kHz audio
    python level_sweep.py --calls-dir callhome_out --root ... --checkpoint finetune_out\\best.pth
    python level_sweep.py --calls-dir callhome_out --root ... --quick
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SR = 16000
N = 64600


def scale_to(x, target_db):
    """one static gain so the clip's 95th-percentile frame level equals target_db (limited to +-40 dB)."""
    from levelnorm import p95_db
    cur = p95_db(x)
    if not np.isfinite(cur):
        return x
    g = float(np.clip(target_db - cur, -40.0, 40.0))
    return np.clip(x * 10 ** (g / 20), -1.0, 1.0).astype(np.float32)


def crop_or_tile(x):
    return x[:N] if len(x) >= N else np.tile(x, int(np.ceil(N / len(x))))[:N]


def p_spoof(model, dev, arrs, bs=8):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), bs):
            b = torch.from_numpy(np.stack([crop_or_tile(np.asarray(a, np.float32)) for a in arrs[i:i + bs]])).to(dev)
            o = model(b)
            o = o[1] if isinstance(o, (tuple, list)) else o
            out += torch.softmax(o, dim=1)[:, 0].cpu().tolist()
    return np.array(out)


def load_calls(directory, wanted, max_s, hop_s):
    paths = sorted(glob.glob(os.path.join(directory, "call*_mix.wav")))
    calls = []
    for p in paths:
        c = int(re.search(r"call(\d+)_mix", p).group(1))
        if wanted and c not in wanted:
            continue
        x, sr = sf.read(p, frames=int(max_s * SR), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if len(x) >= N:
            calls.append((c, x, [s for s in range(0, len(x) - N + 1, int(hop_s * SR))]))
    return calls


def load_la(root, n, seed, phone, protocols):
    from silence_experiment import load_labels, phone8k
    labels = load_labels(root, protocols)
    files = sorted(glob.glob(os.path.join(root, "**", "*.flac"), recursive=True))
    items = [(f, labels[os.path.splitext(os.path.basename(f))[0]]) for f in files if os.path.splitext(os.path.basename(f))[0] in labels]
    rng = np.random.default_rng(seed)
    picked = []
    for key in ("bonafide", "spoof"):
        sel = [it for it in items if it[1] == key]
        rng.shuffle(sel)
        picked += sel[: n // 2]
    clips, labs = [], []
    for f, key in picked:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        clips.append(phone8k(x) if phone else x)
        labs.append(1 if key == "bonafide" else 0)
    return clips, np.array(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls-dir", default="callhome_out")
    ap.add_argument("--calls", type=int, nargs="+", help="call numbers (default all)")
    ap.add_argument("--root", required=True, help="ASVspoof 2019 LA folder (flac + protocols), as for silence_experiment.py")
    ap.add_argument("--protocol", action="append", default=[],
                    help="protocol .txt with the real/fake labels (repeatable); default: ASVspoof2019.LA.cm.*.txt found under --root")
    ap.add_argument("--targets", type=float, nargs="+", default=[-34, -28, -22, -16, -10], help="p95 levels in dB to test")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--hop", type=float, default=3.0, help="seconds between scored 4 s windows of a call")
    ap.add_argument("--la-n", type=int, default=100, help="LA clips (half real, half fake when available)")
    ap.add_argument("--la-phone", action="store_true", help="resample the LA clips to 8 kHz first (so they look like calls)")
    ap.add_argument("--checkpoint", help="fine-tuned AASIST weights")
    ap.add_argument("--quick", action="store_true", help="3 targets, 30 s per call, 40 LA clips")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="level_sweep.csv")
    args = ap.parse_args()
    if args.quick:
        args.targets, args.max_seconds, args.la_n = [-30, -20, -10], 30.0, 40

    import aasist_io
    from silence_experiment import compute_eer
    model, dev = aasist_io.load(args.checkpoint)
    calls = load_calls(args.calls_dir, args.calls, args.max_seconds, args.hop)
    if not calls:
        sys.exit(f"No usable callNNN_mix.wav (>= 4 s) in {args.calls_dir}")
    la, la_lab = load_la(args.root, args.la_n, args.seed, args.la_phone, args.protocol)
    nb = int(la_lab.sum())
    if len(la) == 0:
        sys.exit("No labeled LA clips found: check --root (needs the flac files and the ASVspoof2019.LA.cm.*.txt protocol)")
    conds = [None] + list(args.targets)
    n_win = sum(len(w) for _, _, w in calls)
    print(f"model: {args.checkpoint or 'AASIST (pretrained)'}   device: {dev}")
    print(f"{len(calls)} calls, {n_win} windows each condition; LA: {len(la)} clips ({nb} real, {len(la) - nb} fake)"
          f"{', 8 kHz' if args.la_phone else ''};  {len(conds)} conditions -> about {len(conds) * (n_win + len(la))} windows\n", flush=True)
    if nb < 10:
        print("Warning: few real LA clips: the LA error figures are rough.\n")

    rows = []
    percall = {}
    for t in conds:
        name = "natural" if t is None else f"{t:+.0f} dB"
        fa = []
        for c, x, starts in calls:
            y = x if t is None else scale_to(x, t)
            p = p_spoof(model, dev, [y[s:s + N] for s in starts])
            fa.append(float(np.mean(p > 0.5)))
        percall[name] = fa
        ys = [c if t is None else scale_to(c, t) for c in la]
        p = p_spoof(model, dev, ys)
        real_flagged = float(np.mean(p[la_lab == 1] > 0.5)) if nb else float("nan")
        fake_caught = float(np.mean(p[la_lab == 0] > 0.5)) if (la_lab == 0).any() else float("nan")
        eer = 100 * compute_eer(la_lab, 1.0 - p)
        rows.append({"level": name, "callhome_mean_fa": float(np.mean(fa)), "callhome_worst_fa": float(np.max(fa)),
                     "la_real_flagged": real_flagged, "la_fake_caught": fake_caught, "la_eer": eer})
        r = rows[-1]
        print(f"  done {name:>8}: CallHome {100 * r['callhome_mean_fa']:.0f}% (worst {100 * r['callhome_worst_fa']:.0f}%) | "
              f"LA real flagged {100 * real_flagged:.0f}%  fake caught {100 * fake_caught:.0f}%  EER {eer:.1f}%", flush=True)

    print(f"\n{'level':>9} | {'CallHome windows flagged (real)':>33} | {'LA real flagged':>16} {'LA fake caught':>15} {'LA EER':>7}")
    print(f"{'':>9} | {'mean of calls':>16} {'worst call':>16} |")
    for r in rows:
        print(f"{r['level']:>9} | {100 * r['callhome_mean_fa']:>15.0f}% {100 * r['callhome_worst_fa']:>15.0f}% | "
              f"{100 * r['la_real_flagged']:>15.0f}% {100 * r['la_fake_caught']:>14.0f}% {r['la_eer']:>6.1f}%")
    print("\nper call (share of windows scored fake):")
    print(f"{'level':>9} | " + " ".join(f"call{c:03d}" for c, _, _ in calls))
    for name, fa in percall.items():
        print(f"{name:>9} | " + " ".join(f"{100 * v:>7.0f}%" for v in fa))

    ok = [r for r in rows if r["la_fake_caught"] >= 0.90 and r["la_real_flagged"] <= 0.10]
    print("\nLevels where LA fakes are still caught (>= 90%) and LA real clips are rarely flagged (<= 10%):",
          ", ".join(r["level"] for r in ok) if ok else "none")
    if ok:
        b = min(ok, key=lambda r: r["callhome_mean_fa"])
        nat = rows[0]["callhome_mean_fa"]
        print(f"Of those, the fewest CallHome false alarms: {b['level']} ({100 * b['callhome_mean_fa']:.0f}% of windows; "
              f"natural level: {100 * nat:.0f}%).")
        if b["callhome_mean_fa"] > 0.5 * nat:
            print("That is not much better than the natural level: loudness alone does not fix the false alarms -> "
                  "fine-tuning (random-gain + phone augmentation) is the next step.")
    else:
        print("No single level gives both: loudness alone does not fix the false alarms -> fine-tuning is the next step.")
    print("\nCaveats: one static gain per call or clip (a streaming normaliser is worse); few calls; LA fakes are the 2019 attacks.")
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()