"""Run: python tests/test_level_sweep.py   (numpy only)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from levelnorm import p95_db
from level_sweep import scale_to, crop_or_tile, N
from test_quality import wideband


def test_scale_hits_the_target_level_both_ways():
    x = wideband(6, 1)
    for t in (-34.0, -22.0, -10.0):
        assert abs(p95_db(scale_to(x, t)) - t) < 0.6, t


def test_scale_is_safe_for_silence_and_does_not_clip():
    assert np.all(scale_to(np.zeros(16000, np.float32), -20.0) == 0)
    assert np.max(np.abs(scale_to(wideband(4, 2) * 30, -3.0))) <= 1.0


def test_crop_or_tile_gives_4s():
    assert len(crop_or_tile(np.ones(1000, np.float32))) == N
    assert len(crop_or_tile(np.ones(N + 5000, np.float32))) == N


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)