"""Apply the silence_experiment conditions to ONE recording, scored over sliding 4 s windows."""
import argparse
import numpy as np
import detectors
from capture import load_audio_16k
from silence_experiment import transform, CONDITIONS, SR, WIN
def window_scores(model, device, y):
    y = np.asarray(y, dtype=np.float32)
    starts = [0] if y.size <= WIN else list(range(0, y.size - WIN + 1, SR))
    return np.array([detectors.spoof_prob_window(model, device, y[s:s + WIN]) for s in starts])
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default="rec.aac")
    ap.add_argument("--model", choices=["aasist", "ssl"], default="aasist")
    args = ap.parse_args()
    x = np.asarray(load_audio_16k(args.file), dtype=np.float32)
    model, device = detectors.load_model(args.model)
    rng = np.random.default_rng(42)
    print(f"{args.file}  {x.size / SR:.1f}s  model={args.model}\n")
    print(f"{'condition':<12}{'dur_s':>7}{'windows':>9}{'mean':>7}{'median':>8}{'max':>7}{'>0.5':>7}")
    detail = []
    for c in CONDITIONS:
        y = transform(c, x, rng)
        if y.size < SR // 4:
            y = x
        s = window_scores(model, device, y)
        print(f"{c:<12}{y.size / SR:>7.1f}{s.size:>9}{s.mean():>7.2f}{np.median(s):>8.2f}{s.max():>7.2f}{(s > 0.5).mean():>7.0%}")
        detail.append((c, s))
    print("\nper-window scores:")
    for c, s in detail:
        print(f"{c:<12}" + " ".join(f"{v:.2f}" for v in s))
if __name__ == "__main__":
    main()
