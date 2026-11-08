"""Why does a REAL recording score as fake?  Scores one file several ways with the chosen detector.

    python diagnose_spoof.py rec.aac            (SPOOF_MODEL=ssl python diagnose_spoof.py rec.aac)

Prints level stats and the mean spoof probability for: the raw audio, peak-normalised audio,
speech-only audio, and the first / last 4 s. If every variant stays near 1.0, the problem is the
model/domain mismatch (microphone, AAC, noise), not a level or silence problem.
"""
import sys

import numpy as np

import detectors
from capture import load_audio_16k
from live import EnergyVAD

SR = 16000


def mean_p(model, device, wav):
    n = detectors.MAX_LEN
    wins = [wav[i:i + n] for i in range(0, max(1, len(wav) - n + 1), SR)] or [wav]
    return float(np.mean([detectors.spoof_prob_window(model, device, w) for w in wins]))


def main():
    wav = load_audio_16k(sys.argv[1] if len(sys.argv) > 1 else "rec.aac")
    model, device = detectors.load_model()
    peak, rms = float(np.max(np.abs(wav))), float(np.sqrt(np.mean(wav ** 2)))
    print(f"{len(wav) / SR:.1f} s | peak {peak:.3f} | rms {20 * np.log10(rms + 1e-9):.1f} dBFS")
    speech = EnergyVAD().process(wav)
    rows = [("raw", wav), ("peak-normalised", wav / (peak + 1e-9) * 0.9),
            ("speech only", speech if len(speech) > SR else wav),
            ("first 4 s", wav[:4 * SR]), ("last 4 s", wav[-4 * SR:])]
    for name, x in rows:
        print(f"{name:16s} mean spoof prob = {mean_p(model, device, x):.3f}")


if __name__ == "__main__":
    main()
