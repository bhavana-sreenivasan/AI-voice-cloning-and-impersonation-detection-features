"""Run: python tests/test_augment.py   (numpy + scipy)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import augment, quality
from test_quality import wideband
from levelnorm import frame_db

SR = 16000


def clip(seed=0, sec=5):
    return wideband(sec, seed)


def test_shapes_range_and_finite():
    rng = np.random.default_rng(0)
    for i in range(30):
        y = augment.augment(clip(i % 5), rng)
        assert y.shape == (augment.N,) and y.dtype == np.float32
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0


def test_trim_and_pad_and_nopause():
    x = np.concatenate([np.zeros(SR, np.float32) + 1e-4, clip(1, 3), np.zeros(SR, np.float32) + 1e-4])
    t = augment.trim_edges(x)
    assert len(t) < len(x) - SR                              # most of the 2 s of edge silence is gone
    p = augment.pad_noise(t, np.random.default_rng(0), 1.0, 0.5)
    assert abs(len(p) - (len(t) + int(1.5 * SR))) <= 1
    assert len(augment.no_pauses(clip(2))) <= len(clip(2))


def test_phone_channel_is_narrowband_even_with_noise_and_mulaw():
    rng = np.random.default_rng(3)
    flagged = sum(quality.check(augment.phone_channel(clip(i % 4, 6), rng))["narrowband"] for i in range(30))
    assert flagged >= 27, flagged                           # the chain never leaks energy above 4 kHz
    assert not quality.check(clip(0, 6))["narrowband"]       # input itself is wideband


def test_gain_range_and_clip_protection():
    x = clip(0)
    lo = 10 * np.log10(np.mean(augment.gain_db(x, -20) ** 2) / np.mean(x ** 2))
    assert abs(lo + 20) < 0.2
    assert np.max(np.abs(augment.gain_db(x * 20, 20))) <= 1.0


def test_short_clip_is_tiled_and_augment_is_reproducible():
    short = clip(0, 1)[: SR // 2]
    y = augment.crop_or_tile(short)
    assert len(y) == augment.N
    a = augment.augment(clip(1), np.random.default_rng(7))
    b = augment.augment(clip(1), np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_label_independent_edge_silence_after_equalisation():
    """a clip with 2 s of edge silence and one without end up with similar edge lengths on average."""
    rng = np.random.default_rng(0)
    core = clip(1, 3)
    with_edges = np.concatenate([np.zeros(2 * SR, np.float32) + 1e-4, core, np.zeros(2 * SR, np.float32) + 1e-4])
    def edge_s(y):
        d = frame_db(y, 320); q = d < np.percentile(d, 95) - 35; idx = np.where(~q)[0]
        return (idx[0] + len(d) - 1 - idx[-1]) * 320 / SR
    a = np.mean([edge_s(augment.augment(with_edges, rng, p=dict(phone=0, gain=0, nopause=0, trim=1.0))) for _ in range(40)])
    b = np.mean([edge_s(augment.augment(core, rng, p=dict(phone=0, gain=0, nopause=0, trim=1.0))) for _ in range(40)])
    assert abs(a - b) < 0.4, (a, b)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)