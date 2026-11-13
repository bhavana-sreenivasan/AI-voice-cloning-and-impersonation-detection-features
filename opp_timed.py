import argparse, csv, os
import numpy as np
from eval_callhome import load_wav, run_analyzer, SR
import detectors
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="callhome_out")
ap.add_argument("--calls", type=int, nargs="+", required=True)
ap.add_argument("--max-seconds", type=float, default=120.0)
ap.add_argument("--user", default="A")
args = ap.parse_args()
model, device = detectors.load_model("aasist")
score = lambda w: detectors.spoof_prob_window(model, device, w)
print(f"{'call':>5}{'input':>16}{'windows':>9}{'>0.5':>7}{'mean p':>8}{'alert':>8}")
for c in args.calls:
    mix = load_wav(os.path.join(args.dir, f"call{c:03d}_mix.wav"), args.max_seconds)
    segs = [(float(r["start_s"]), float(r["end_s"]), r["speaker"])
            for r in csv.DictReader(open(os.path.join(args.dir, f"call{c:03d}_segments.csv"), newline=""))]
    fr = mix[: len(mix) // 320 * 320].reshape(-1, 320)
    db = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-10)
    amp = 10 ** (float(np.clip(np.percentile(db, 10), -70, -45)) / 20)
    rng = np.random.default_rng(0)
    timed = mix.copy()
    for s, e, spk in segs:
        if spk == args.user:
            a, b = int(s * SR), min(len(mix), int(e * SR))
            if b > a:
                timed[a:b] = (amp * rng.standard_normal(b - a)).astype(np.float32)
    for name, x in (("mix", mix), ("opponent_timed", timed)):
        r = run_analyzer(x, score, False, 3)
        print(f"{c:>5}{name:>16}{r['n']:>9}{100 * np.mean(r['p'] > 0.5):>6.0f}%{np.mean(r['p']):>8.2f}{'ALERT' if r['alert'] else '-':>8}", flush=True)
