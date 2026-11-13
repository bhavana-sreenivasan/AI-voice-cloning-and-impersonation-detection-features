"""augment.py - training augmentations aimed at the shortcuts the experiments found.

Findings these address (see silence_experiment.py / the CallHome runs):
  * the detectors read silence: real clips have long edges and many pauses, fakes are trimmed
        -> trim_edges + pad_random + no_pauses are applied to BOTH classes, so silence stops
           being a label
  * the score moves with loudness (level_sweep.py: CallHome false alarms 3% -> 92% from -34 to -10 dB, and LA fakes
    caught 12% -> 96%)                                   -> random_level: every clip is set to a RANDOM TARGET
    level, so loudness carries no label and no corpus information (LA clips sit near -12 dB, calls near -26 dB;
    a random relative gain would keep that ~13 dB gap)
  * phone audio (8 kHz, codec, noise, gating) is out of distribution -> phone_channel
All functions take / return float32 mono 16 kHz arrays.  numpy + scipy only.
"""
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

from levelnorm import frame_db

SR = 16000
N = 64600            # 4 s: the model input
FR = 320             # 20 ms
REL_QUIET_DB = 35.0

DEFAULT_P = dict(trim=0.7, pad=0.7, nopause=0.2, phone=0.6, gain=1.0)   # 'gain' = probability of the random target level


def _quiet(x):
    d = frame_db(x, FR)
    if d.size == 0:
        return d, np.zeros(0, dtype=bool)
    return d, d < (np.percentile(d, 95) - REL_QUIET_DB)


def floor_db(x):
    d = frame_db(x, FR)
    return float(np.clip(np.percentile(d, 10), -70.0, -45.0)) if d.size else -60.0


def trim_edges(x):
    d, q = _quiet(x)
    idx = np.where(~q)[0]
    return x if idx.size == 0 else x[idx[0] * FR:(idx[-1] + 1) * FR]


def no_pauses(x):
    d, q = _quiet(x)
    if d.size == 0 or q.all():
        return x
    return x[: d.size * FR].reshape(d.size, FR)[~q].reshape(-1)


def pad_noise(x, rng, left_s, right_s):
    """faint room noise (the clip's own noise floor) before and after the clip."""
    lvl = 10 ** (floor_db(x) / 20)
    mk = lambda s: (lvl * rng.standard_normal(int(s * SR))).astype(np.float32)   # noqa: E731
    return np.concatenate([mk(left_s), x, mk(right_s)])


def pad_random(x, rng, max_s=1.5):
    return pad_noise(x, rng, rng.uniform(0, max_s), rng.uniform(0, max_s))


def gain_db(x, g):
    return np.clip(x * 10 ** (g / 20), -1.0, 1.0).astype(np.float32)


def random_gain(x, rng, lo=-25.0, hi=10.0):
    return gain_db(x, rng.uniform(lo, hi))


def random_level(x, rng, lo=-38.0, hi=-8.0):
    """scale the clip so its 95th-percentile frame level is a random value in [lo, hi] dB (limited to +-45 dB of gain)."""
    from levelnorm import p95_db
    cur = p95_db(x)
    if not np.isfinite(cur):
        return x
    return gain_db(x, float(np.clip(rng.uniform(lo, hi) - cur, -45.0, 45.0)))


def phone8k(x):
    """recorded at 8 kHz: nothing above 4 kHz (resampled back to 16 kHz)."""
    n = len(x)
    y = resample_poly(resample_poly(x, 1, 2), 2, 1).astype(np.float32)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def phone_channel(x, rng):
    """8 kHz telephone-like chain, all stages at 8 kHz so no energy appears above 4 kHz:
    noise -> band-pass 300-3400 Hz -> mu-law 8 bit -> optional digital-silence gating."""
    n = len(x)
    y = resample_poly(x, 1, 2).astype(np.float32)
    if rng.random() < 0.5:                                           # background noise, 15-35 dB SNR
        snr = rng.uniform(15, 35)
        y = y + (np.sqrt(np.mean(y ** 2) + 1e-12) * 10 ** (-snr / 20) * rng.standard_normal(len(y))).astype(np.float32)
    if rng.random() < 0.6:                                           # telephone band
        y = sosfilt(butter(4, [300, 3400], btype="band", fs=8000, output="sos"), y).astype(np.float32)
    if rng.random() < 0.5:                                           # G.711-like mu-law quantisation
        mu = 255.0
        v = np.clip(y, -1, 1)
        c = np.sign(v) * np.log1p(mu * np.abs(v)) / np.log1p(mu)
        c = np.round((c + 1) * 127.5) / 127.5 - 1
        y = (np.sign(c) * (np.expm1(np.abs(c) * np.log1p(mu)) / mu)).astype(np.float32)
    if rng.random() < 0.3:                                           # noise-gate: digital silence between words
        fr = 160
        m = len(y) // fr
        if m:
            lv = 10 * np.log10(np.mean(y[: m * fr].reshape(m, fr) ** 2, axis=1) + 1e-12)
            gate = lv < (np.percentile(lv, 95) - 40.0)
            f = y[: m * fr].reshape(m, fr)
            f[gate] = 0.0
            y[: m * fr] = f.reshape(-1)
    out = resample_poly(y, 2, 1).astype(np.float32)
    return out[:n] if len(out) >= n else np.pad(out, (0, n - len(out)))


def crop_or_tile(x, rng=None, n=N):
    """random 4 s crop of a long clip; short clips are tiled (as the original training code does)."""
    if len(x) >= n:
        s = 0 if rng is None else int(rng.integers(0, len(x) - n + 1))
        return x[s:s + n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def augment(x, rng, already_phone=False, p=None):
    """silence equalisation -> phone channel -> random target level -> 4 s crop.  Label-independent by design."""
    p = {**DEFAULT_P, **(p or {})}
    x = np.asarray(x, dtype=np.float32)
    orig = x
    if rng.random() < p["trim"]:
        x = trim_edges(x)
    if rng.random() < p["pad"]:
        x = pad_random(x, rng)
    if rng.random() < p["nopause"]:
        x = no_pauses(x)
    if len(x) < SR:                                  # too little left: fall back to the original clip
        x = orig
    if not already_phone and rng.random() < p["phone"]:
        x = phone_channel(x, rng)
    if rng.random() < p["gain"]:
        x = random_level(x, rng)
    return np.ascontiguousarray(crop_or_tile(x, rng), dtype=np.float32)