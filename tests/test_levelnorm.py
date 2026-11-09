"""Run: python tests/test_levelnorm.py   (needs only numpy; run from the project folder)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from levelnorm import LevelNormalizer

SR = 16000


def fake_speech(seconds=12, seed=0):
    """Syllable-like bursts (4 Hz) with pauses, ~ -30 dB RMS, plus a faint noise floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2
    env *= (np.sin(2 * np.pi * 0.35 * t) > -0.2)                   # slow pauses
    x = env * rng.standard_normal(t.size) * 0.05 + 0.0005 * rng.standard_normal(t.size)
    return x.astype(np.float32)


def speech_rms_db(x):
    from levelnorm import p95_db
    return p95_db(x)                                              # level of the loud frames


def run(x, chunk=SR // 2):
    ln = LevelNormalizer()
    out = [ln.process(x[i:i + chunk]) for i in range(0, x.size, chunk)]
    return np.concatenate(out)


def test_length_and_range_unchanged():
    x = fake_speech(6)
    y = run(x)
    assert y.shape == x.shape and np.max(np.abs(y)) <= 1.0


def test_gain_invariance_within_2db():
    base = fake_speech(14)
    levels = []
    for g in (0.25, 0.5, 1.0, 2.0, 4.0):
        y = run(np.clip(base * g, -1, 1))
        levels.append(speech_rms_db(y[6 * SR:]))                   # after settling
    spread = max(levels) - min(levels)
    assert spread < 2.0, f"level spread {spread:.2f} dB: {np.round(levels, 1)}"


def test_silence_and_noise_are_not_boosted_to_speech_level():
    n = (0.0005 * np.random.default_rng(1).standard_normal(SR * 4)).astype(np.float32)
    y = run(n)
    assert np.sqrt(np.mean(y * y)) < 0.02                          # still quiet


def test_all_zero_input_is_safe():
    y = run(np.zeros(SR * 2, dtype=np.float32))
    assert np.all(y == 0)


def test_gain_is_capped():
    ln = LevelNormalizer(max_gain_db=18.0)
    quiet = fake_speech(6) * 0.01                                  # about -50 dB speech
    for i in range(0, quiet.size, SR // 2):
        ln.process(quiet[i:i + SR // 2])
    assert ln.gain_db <= 18.0 + 1e-6


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
