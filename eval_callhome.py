"""eval_callhome.py - false alarms of the spoof detector on REAL phone calls (CallHome), many calls at once.

Every CallHome call is real, so every window the model calls fake and every alert is a false alarm.
For each call (made by callhome_extract.py) it scores two inputs with the SAME alert logic as live_cli.py
(LiveAnalyzer: 4 s window, 1 s hop, median of 3, alert after 3 windows > 0.7 in a row):
  mix        callNNN_mix.wav       both speakers, natural pauses (what a call recording contains)
  opponent   callNNN_opponent.wav  the true other speaker alone, non-overlapping talk glued together
                                   (pauses removed: expect HIGHER fake scores, see silence_experiment.py)
and reports, with the narrowband gate OFF (raw detector) and ON (as shipped):
  windows >0.5   share of windows scored fake
  alert          did an alert fire at any point
With --similarity it also measures, with the ground-truth turns in callNNN_segments.csv, how well the speaker
similarity (speaker.py, your SPEAKER_BACKEND) separates the user from the opponent: AUC + best --drop-above.

Compare models by running it twice:
    python eval_callhome.py --dir callhome_out                                  # pretrained AASIST
    python eval_callhome.py --dir callhome_out --checkpoint finetune_out\\best.pth   # fine-tuned
    python eval_callhome.py --dir callhome_out --model ssl                      # SSL-AASIST (needs the fairseq env)
Use --calls to restrict (e.g. only the calls held out from fine-tuning: --calls 3 4):
    python eval_callhome.py --dir callhome_out --calls 3 4 --max-seconds 120 --similarity
"""
import argparse
import csv
import glob
import hashlib
import os
import re
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SR = 16000
FIXED_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45)


def load_wav(path, max_s):
    x, sr = sf.read(path, frames=int(max_s * SR) if max_s else -1, dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: expected 16 kHz"
    return x.mean(axis=1)


def run_analyzer(x, score, gate, smooth):
    from live import LiveAnalyzer
    an = LiveAnalyzer(score, min_s=2.0, hop_s=1.0, smooth=smooth, check_quality=gate)
    for i in range(0, len(x), SR // 2):
        an.push(x[i:i + SR // 2])
    ev = an.history
    return {
        "n": len(ev),
        "p": np.array([e["p_raw"] for e in ev]),
        "alert": any(e["new_alert"] for e in ev),
        "first": next((e["t"] for e in ev if e["new_alert"]), None),
        "unrel": float(np.mean([e["unreliable"] for e in ev])) if ev else 0.0,
    }


def similarity_report(call_dir, idx, mix, segs_path, user, max_s):
    """-> (auc, best_threshold, n_user, n_opp, user_sims, opp_sims) for one call, or None if too few clean slices."""
    import torch
    import speaker
    from me_similarity import auc, slice_label
    user_wav = os.path.join(call_dir, f"call{idx:03d}_user.wav")
    if not (os.path.isfile(user_wav) and os.path.isfile(segs_path)):
        return None
    emb = lambda w: speaker.embed(torch.tensor(np.asarray(w, np.float32))).numpy()   # noqa: E731
    ref = emb(load_wav(user_wav, 30))
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    with open(segs_path, newline="") as fh:
        segs = [(float(r["start_s"]), float(r["end_s"]), r["speaker"]) for r in csv.DictReader(fh)]
    groups = {"user": [], "opponent": []}
    for i in range(0, len(mix) - SR + 1, SR):
        g = slice_label(segs, user, i / SR, (i + SR) / SR)
        if g and np.sqrt(np.mean(mix[i:i + SR] ** 2)) > 1e-3:
            e = emb(mix[i:i + SR])
            groups[g].append(float(ref @ (e / (np.linalg.norm(e) + 1e-12))))
    gu, go = np.array(groups["user"]), np.array(groups["opponent"])
    if len(gu) < 5 or len(go) < 5:
        return None
    cand = np.unique(np.concatenate([gu, go]))
    bal, t = max((0.5 * (np.mean(gu >= t) + np.mean(go < t)), t) for t in cand)
    return auc(gu, go), float(t), len(gu), len(go), gu, go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="callhome_out", help="folder made by callhome_extract.py")
    ap.add_argument("--calls", type=int, nargs="+", help="call numbers to use (default: all found)")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="use at most this much of each call")
    ap.add_argument("--model", choices=["aasist", "ssl"], default="aasist")
    ap.add_argument("--checkpoint", help="fine-tuned AASIST weights (.pth), --model aasist only")
    ap.add_argument("--smooth", type=int, default=3, help="median over the last N window scores (live_cli default 3)")
    ap.add_argument("--similarity", action="store_true", help="also evaluate speaker similarity with the ground-truth turns")
    ap.add_argument("--user", default="A", help="speaker label of the user (default A)")
    ap.add_argument("--csv", default="callhome_eval.csv")
    args = ap.parse_args()

    found = sorted(int(re.search(r"call(\d+)_mix", p).group(1)) for p in glob.glob(os.path.join(args.dir, "call*_mix.wav")))
    calls = [c for c in found if (not args.calls or c in args.calls)]
    if not calls:
        sys.exit(f"No callNNN_mix.wav found in {args.dir} for calls {args.calls or 'all'}")

    import detectors
    if args.checkpoint:
        if args.model != "aasist":
            sys.exit("--checkpoint is only for --model aasist")
        import aasist_io
        model, device = aasist_io.load(args.checkpoint)
    else:
        model, device = detectors.load_model(args.model)
    tag = args.checkpoint or f"{args.model} (pretrained)"
    print(f"model: {tag}   device: {device}   calls: {calls}   up to {args.max_seconds:.0f}s each\n")

    cache = {}

    def score(w):
        key = (len(w), hashlib.blake2b(np.ascontiguousarray(w).tobytes(), digest_size=8).digest())
        if key not in cache:                       # the gate-on run replays the same windows: no second inference
            cache[key] = detectors.spoof_prob_window(model, device, w)
        return cache[key]

    rows = []
    import quality
    from levelnorm import p95_db
    print(f"{'call':>5} {'input':<9}{'windows':>8}{'>0.5':>7}{'mean p':>8}{'alert (raw)':>13}{'alert (gate)':>14}{'gate flags':>12}"
          f"{'level dB':>10}{'4.5-7.5k vs 1-3.4k':>20}")
    for c in calls:
        for name in ("mix", "opponent"):
            path = os.path.join(args.dir, f"call{c:03d}_{name}.wav")
            if not os.path.isfile(path):
                continue
            x = load_wav(path, args.max_seconds)
            if len(x) < 2 * SR:
                continue
            raw = run_analyzer(x, score, False, args.smooth)
            gated = run_analyzer(x, score, True, args.smooth)
            r = {"call": c, "input": name, "windows": raw["n"], "frac_gt05": float(np.mean(raw["p"] > 0.5)),
                 "mean_p": float(np.mean(raw["p"])), "peak_p": float(np.max(raw["p"])),
                 "alert_raw": raw["alert"], "alert_gate": gated["alert"], "gate_flags": gated["unrel"],
                 "p95_db": p95_db(x), "hf_db": quality.hf_ratio_db(x)}
            rows.append(r)
            print(f"{c:>5} {name:<9}{r['windows']:>8}{100 * r['frac_gt05']:>6.0f}%{r['mean_p']:>8.2f}"
                  f"{'ALERT' if r['alert_raw'] else '-':>13}{'ALERT' if r['alert_gate'] else '-':>14}{100 * r['gate_flags']:>11.0f}%"
                  f"{r['p95_db']:>10.1f}{('n/a' if r['hf_db'] is None else format(r['hf_db'], '.1f')):>20}", flush=True)

    sim_rows = []
    if args.similarity:
        import speaker
        print(f"\nspeaker similarity, backend={speaker.BACKEND}, ground-truth turns:")
        for c in calls:
            mixp = os.path.join(args.dir, f"call{c:03d}_mix.wav")
            r = similarity_report(args.dir, c, load_wav(mixp, args.max_seconds), os.path.join(args.dir, f"call{c:03d}_segments.csv"),
                                  args.user, args.max_seconds)
            if r:
                sim_rows.append((c, *r))
                print(f"  call{c:03d}: AUC {r[0]:.2f}   best --drop-above {r[1]:.2f}   ({r[2]} user / {r[3]} opponent clean slices)", flush=True)
            else:
                print(f"  call{c:03d}: too few clean slices (or files missing)")

    if sim_rows:
        print("\none fixed --drop-above for every call (want both columns high; 'worst call' = lowest balanced accuracy):")
        print(f"  {'threshold':>9} {'user slices dropped':>20} {'opponent slices kept':>22} {'worst call':>11}")
        gu_all = np.concatenate([s[5] for s in sim_rows])
        go_all = np.concatenate([s[6] for s in sim_rows])
        for t in FIXED_THRESHOLDS:
            worst = min(0.5 * (np.mean(s[5] >= t) + np.mean(s[6] < t)) for s in sim_rows)
            print(f"  {t:9.2f} {100 * np.mean(gu_all >= t):19.0f}% {100 * np.mean(go_all < t):21.0f}% {100 * worst:10.0f}%")

    with open(args.csv, "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), restval="")
            w.writeheader()
            w.writerows(rows)

    print(f"\n===== SUMMARY  ({tag}) =====")
    for name in ("mix", "opponent"):
        rr = [r for r in rows if r["input"] == name]
        if not rr:
            continue
        print(f"{name:<9} {len(rr)} calls | windows scored fake: mean {100 * np.mean([r['frac_gt05'] for r in rr]):.0f}% "
              f"(range {100 * min(r['frac_gt05'] for r in rr):.0f}-{100 * max(r['frac_gt05'] for r in rr):.0f}%) | "
              f"calls with an alert: raw {sum(r['alert_raw'] for r in rr)}/{len(rr)}, with the gate {sum(r['alert_gate'] for r in rr)}/{len(rr)}")
    mixr = [r for r in rows if r["input"] == "mix"]
    if len(mixr) >= 4:
        f = np.array([r["frac_gt05"] for r in mixr])
        for key, label in (("p95_db", "level"), ("hf_db", "bandwidth")):
            v = np.array([np.nan if r[key] is None else r[key] for r in mixr])
            ok = ~np.isnan(v)
            if ok.sum() >= 4 and v[ok].std() > 0 and f[ok].std() > 0:
                print(f"across {int(ok.sum())} calls (mix): correlation of {label} with the share of windows scored fake "
                      f"r = {np.corrcoef(v[ok], f[ok])[0, 1]:+.2f}  (few calls: only a hint)")
    if sim_rows:
        a = np.array([s[1] for s in sim_rows])
        t = np.array([s[2] for s in sim_rows])
        print(f"similarity: AUC median {np.median(a):.2f} (min {a.min():.2f}); best --drop-above median {np.median(t):.2f} "
              f"(range {t.min():.2f}-{t.max():.2f}) over {len(sim_rows)} calls")
    print("All CallHome audio is real: every 'fake' window or alert above is a false alarm.")
    print(f"Per-call numbers: {args.csv}")


if __name__ == "__main__":
    main()