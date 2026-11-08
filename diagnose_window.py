"""Which parts of a recording look fake, and does silence / loudness drive the score?

    python diagnose_windows.py rec.aac          (SPOOF_MODEL=ssl python diagnose_windows.py rec.aac)

1) Table: every 4 s window (1 s hop) with its level, % speech frames and spoof probability.
2) Variants of the whole file: raw, speech-only, speech-only with short pauses put back,
   and different gains. If "speech-only" is much higher than "raw", or putting pauses back
   lowers it again, the model is leaning on silence. If the gains differ a lot, it leans on level.
"""
import sys

import numpy as np

import detectors
from capture import load_audio_16k
from live import EnergyVAD

SR = 16000
N = detectors.MAX_LEN


def windows(x):
    if len(x) <= N:
        return [(0, x)]
    return [(s, x[s:s + N]) for s in range(0, len(x) - N + 1, SR)]


def mean_p(model, device, x):
    return float(np.mean([detectors.spoof_prob_window(model, device, w) for _, w in windows(x)]))


def with_gaps(speech, chunk_s=1.5, gap_s=0.4, level_db=-60.0):
    """Speech-only audio with short faint-noise pauses put back between chunks."""
    rng = np.random.default_rng(0)
    gap = (10 ** (level_db / 20) * rng.standard_normal(int(gap_s * SR))).astype(np.float32)
    parts, c = [], int(chunk_s * SR)
    for i in range(0, len(speech), c):
        parts += [speech[i:i + c], gap]
    return np.concatenate(parts)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "rec.aac"
    wav = load_audio_16k(path)
    model, device = detectors.load_model()
    print(f"{path}: {len(wav) / SR:.1f} s\n")

    print(" start   level(dBFS)  speech%   spoof")
    ps, fr = [], []
    for s, w in windows(wav):
        rms = float(np.sqrt(np.mean(w ** 2)))
        frac = len(EnergyVAD().process(w)) / max(len(w), 1)
        p = detectors.spoof_prob_window(model, device, w)
        ps.append(p); fr.append(frac)
        print(f"{s / SR:6.1f}s   {20 * np.log10(rms + 1e-9):8.1f}   {frac * 100:6.0f}%   {p:6.3f}")
    if len(ps) >= 3 and np.std(fr) > 0 and np.std(ps) > 0:
        print(f"\ncorrelation(speech%, spoof) = {np.corrcoef(fr, ps)[0, 1]:+.2f}   "
              "(strongly positive = more speech/less pause -> looks more fake)")

    speech = EnergyVAD().process(wav)
    print("\nWhole-file variants (mean spoof prob):")
    rows = [("raw", wav), ("speech only (what the live gate does)", speech),
            ("speech only + short pauses put back", with_gaps(speech))]
    for g in (0.5, 1.0, 2.0):
        rows.append((f"raw, gain x{g}", np.clip(wav * g, -1, 1)))
    for name, x in rows:
        print(f"  {name:40s} {mean_p(model, device, x):.3f}")


if __name__ == "__main__":
    main()