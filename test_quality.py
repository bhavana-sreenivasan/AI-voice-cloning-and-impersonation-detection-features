"""Run: python tests/test_quality.py   (numpy + scipy; fuse_risk test is skipped if pipeline can't import)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import quality
from live import LiveAnalyzer

SR = 16000


def wideband(seconds=6, seed=0, fc=1000.0):
    """Speech-shaped noise (-12 dB/oct above ~fc) in 4 Hz bursts."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    X = np.fft.rfft(rng.standard_normal(n))
    X *= 1 / np.sqrt(1 + (np.fft.rfftfreq(n, 1 / SR) / fc) ** 4)
    x = np.fft.irfft(X, n)
    x /= np.sqrt(np.mean(x ** 2))
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 2 + 0.05
    return (0.05 * env * x).astype(np.float32)


def narrowband(seconds=6, seed=0, fc=1000.0):
    """What phone audio looks like to the detectors: recorded at 8 kHz, then resampled to 16 kHz."""
    from scipy.signal import resample_poly
    return resample_poly(resample_poly(wideband(seconds, seed, fc), 1, 2), 2, 1).astype(np.float32)


def test_wideband_is_not_flagged_and_narrowband_is():
    for fc in (500, 1000, 2500, 5000):          # dark ... bright voices
        assert not quality.check(wideband(fc=fc))["narrowband"], fc
        r = quality.check(narrowband(fc=fc))
        assert r["narrowband"] and r["hf_db"] < quality.NARROWBAND_DB, (fc, r)


def test_silence_and_tiny_input_are_unknown_not_flagged():
    assert quality.check(np.zeros(SR))["hf_db"] is None
    assert not quality.check(np.zeros(SR))["narrowband"]
    assert quality.check(np.zeros(100, dtype=np.float32))["hf_db"] is None


def _alerts(x, **kw):
    an = LiveAnalyzer(lambda w: 0.95, min_s=1.0, hop_s=1.0, consec=3, **kw)
    for i in range(0, len(x), SR // 2):
        an.push(x[i:i + SR // 2])
    return an


def test_live_default_unchanged_and_narrowband_suppressed_when_enabled():
    nb = narrowband(8)
    assert any(e["new_alert"] for e in _alerts(nb).history)                   # default: old behaviour
    an = _alerts(nb, check_quality=True)
    assert not any(e["new_alert"] for e in an.history)
    assert all(e["unreliable"] and e["risk"] == "Uncertain" for e in an.history)
    s = an.summary()
    assert s["risk"] == "Uncertain" and s["unreliable_frac"] == 1.0


def test_live_wideband_still_alerts_with_quality_check_on():
    an = _alerts(wideband(8), check_quality=True)
    assert any(e["new_alert"] for e in an.history)
    assert an.summary()["risk"] == "High" and an.summary()["unreliable_frac"] == 0.0


def test_fuse_risk_ignores_unreliable_spoof_but_keeps_identity_signals():
    try:
        import types
        sys.modules.setdefault("librosa", types.ModuleType("librosa"))          # only needed to import pipeline
        from pipeline import fuse_risk
    except ImportError as e:
        print("SKIP fuse_risk test:", e); return
    assert fuse_risk("High", spoof_unreliable=True)[0] == "Uncertain"
    assert fuse_risk("Low", spoof_unreliable=True)[0] == "Uncertain"
    assert fuse_risk("High", "mismatch", spoof_unreliable=True)[0] == "High"
    assert fuse_risk("Low", None, True, spoof_unreliable=True)[0] == "High"     # speaker change
    assert fuse_risk("High")[0] == "High" and fuse_risk("Low")[0] == "Low"      # unchanged by default


def _an(x, **kw):
    an = LiveAnalyzer(lambda w: 0.95, min_s=1.0, hop_s=1.0, consec=3, check_quality=True, **kw)
    for i in range(0, len(x), SR // 2):
        an.push(x[i:i + SR // 2])
    return an


def test_gate_decides_per_call_not_per_window():
    nb = narrowband(30)
    burst = wideband(2, seed=5)                                   # 2 s of wideband audio spliced into a phone call
    x = np.concatenate([nb[:10 * SR], burst, nb[10 * SR:]])
    assert any(e["new_alert"] for e in _an(x, quality_windows=1).history)      # per-window gate: a few windows leak through
    an = _an(x)                                                   # default: median of the last 15 windows
    assert not any(e["new_alert"] for e in an.history)
    assert all(e["unreliable"] for e in an.history)


def test_gate_still_lets_wideband_alert_and_follows_a_channel_change():
    assert any(e["new_alert"] for e in _an(wideband(20)).history)
    x = np.concatenate([wideband(25, 1), narrowband(40, 2)])      # wideband call, then it turns into phone audio
    ev = _an(x).history
    assert not ev[3]["unreliable"] and ev[-1]["unreliable"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)