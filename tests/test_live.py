"""Pure-numpy tests (no torch / models needed):  python -m pytest tests -q   or   python tests/test_live.py"""
import os
import sqlite3
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from capture import ArraySource, mulaw_decode, mulaw_encode, resample_to_16k          # noqa: E402
from calibrate_speaker import suggest_thresholds                                        # noqa: E731,E402
from live import EnergyVAD, GateChain, LiveAnalyzer, PauseLimiter, SessionVoiceTracker, UserVoiceFilter                           # noqa: E402
from caller_pick import pick_caller
from live_session import LiveSession                                                    # noqa: E402

SR = 16000
rng = np.random.default_rng(0)


def speechlike(sec, amp=0.1):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
            + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def silence(sec):
    return (0.0005 * rng.standard_normal(int(sec * SR))).astype(np.float32)


def feed(an, x, chunk=1600):
    ev = []
    for i in range(0, len(x), chunk):
        ev += an.push(x[i:i + chunk])
    return ev


def test_mulaw_roundtrip():
    x = np.clip(0.3 * np.sin(np.linspace(0, 200, 8000)), -1, 1).astype(np.float32)
    y = mulaw_decode(mulaw_encode(x))
    assert len(y) == len(x) and np.max(np.abs(x - y)) < 0.02


def test_resample_length():
    assert abs(len(resample_to_16k(np.zeros(8000, np.float32), 8000)) - 16000) <= 1


def test_vad_drops_silence_keeps_speech():
    v = EnergyVAD()
    assert len(v.process(silence(3))) < 0.1 * 3 * SR
    kept = len(v.process(speechlike(2)))
    assert kept > 0.9 * 2 * SR


def test_events_start_after_min_and_every_hop():
    an = LiveAnalyzer(lambda w: 0.05, min_s=2.0, hop_s=1.0)
    ev = feed(an, speechlike(6))
    assert [e["t"] for e in ev][0] == 2.0 and len(ev) == 5      # t = 2,3,4,5,6
    assert all(e["risk"] == "Low" for e in ev)


def test_alert_after_consecutive_high_and_clears():
    scores = iter([0.1, 0.1] + [0.95] * 5 + [0.05] * 3)
    an = LiveAnalyzer(lambda w: next(scores), min_s=1.0, hop_s=1.0, consec=3)
    ev = feed(an, speechlike(10))
    alerts = [e for e in ev if e["new_alert"]]
    assert len(alerts) == 1 and alerts[0]["t"] == 5.0
    assert ev[-1]["risk"] == "Low" and not ev[-1]["alert_active"]
    assert an.summary()["risk"] == "High"


def test_silence_is_not_scored_with_gate():
    calls = []
    an = LiveAnalyzer(lambda w: calls.append(1) or 0.9, gate=EnergyVAD(), min_s=2.0)
    feed(an, silence(20))
    assert calls == []                                          # remote side silent => nothing scored
    ev = feed(an, speechlike(3))
    assert len(ev) >= 1 and len(calls) >= 1


def test_smoothing_ignores_single_spike():
    scores = iter([0.1, 0.1, 0.99, 0.1, 0.1, 0.1])
    an = LiveAnalyzer(lambda w: next(scores), min_s=1.0, smooth=3, consec=1)
    ev = feed(an, speechlike(6))
    assert not any(e["new_alert"] for e in ev)


def test_slow_scorer_catch_up_scores_once():
    calls = []
    an = LiveAnalyzer(lambda w: calls.append(1) or 0.1, min_s=2.0, hop_s=1.0)
    ev = an.push(speechlike(10))                                # 10 s arrive in one go
    assert len(ev) == 1 and len(calls) == 1 and an.next_eval > an.speech_total


def test_window_is_last_4_seconds():
    sizes = []
    an = LiveAnalyzer(lambda w: sizes.append(len(w)) or 0.1, min_s=2.0)
    feed(an, speechlike(8))
    assert sizes[0] == 2 * SR and max(sizes) == 4 * SR


def test_session_tracker_detects_swap():
    # fake embedding = dominant frequency => voice A (180 Hz) vs voice B (400 Hz)
    def embed(w):
        f = np.abs(np.fft.rfft(w[:SR])); k = int(np.argmax(f))
        v = np.zeros(8); v[0 if k < 300 else 1] = 1.0
        return v + 0.01 * rng.standard_normal(8)
    tr = SessionVoiceTracker(embed, enrol_s=4.0, change_below=0.5, consec=3)
    an = LiveAnalyzer(lambda w: 0.1, tr, min_s=1.0, hop_s=1.0)
    t = np.arange(8 * SR) / SR
    voiceB = (0.1 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    ev = feed(an, speechlike(8)) + feed(an, voiceB)
    assert ev[0]["similarity"] is None                          # still enrolling
    assert any(e["speaker_change"] for e in ev) and an.summary()["speaker_change"]
    assert not any(e["speaker_change"] for e in ev[:6])


def test_threaded_session_end_to_end_and_drops():
    an = LiveAnalyzer(lambda w: 0.1, min_s=1.0)
    sess = LiveSession(an, ArraySource(speechlike(5), realtime=False)).start()
    sess.join(5)
    assert len(sess.events) >= 3 and sess.error is None

    import time
    slow = LiveAnalyzer(lambda w: (time.sleep(0.05), 0.1)[1], min_s=1.0)
    s2 = LiveSession(slow, ArraySource(speechlike(30), realtime=False), max_queue=5).start()
    s2.join(20)
    assert s2.dropped_chunks > 0 and s2.error is None           # fell behind => dropped, didn't hang


def test_threshold_suggestion():
    same = rng.normal(0.7, 0.08, 500); diff = rng.normal(0.05, 0.08, 500)
    r = suggest_thresholds(same, diff)
    assert r["eer"] < 0.01 and 0.1 < r["match_t"] < 0.65 and r["margin"] > 0


def test_call_log_migration_and_model_tag():
    import call_log
    old = tempfile.mktemp(suffix=".db")
    c = sqlite3.connect(old)                                    # a DB created by the OLD schema
    c.execute("""CREATE TABLE calls (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, phone TEXT NOT NULL,
        source TEXT NOT NULL, risk TEXT NOT NULL, label TEXT, p_spoof REAL, identity_name TEXT, identity_score REAL,
        identity_decision TEXT, reason TEXT, audio_sha1 TEXT, verification TEXT, notes TEXT DEFAULT '',
        reported INTEGER DEFAULT 0, report_ref TEXT DEFAULT '', embedding TEXT)""")
    c.execute("INSERT INTO calls (ts,phone,source,risk,embedding) VALUES (1,'+911','upload','High','[1.0,0.0]')")
    c.commit(); c.close()
    call_log.DB = old
    call_log.log_call("+912", "live", "High", embedding=[1.0, 0.0], emb_model="ecapa2")
    assert [r[2] for r in call_log.similar_calls([1.0, 0.0], emb_model="ecapa2")] == ["+912"]
    assert [r[2] for r in call_log.similar_calls([1.0, 0.0], emb_model="speechbrain")] == ["+911"]
    assert len(call_log.similar_calls([1.0, 0.0])) == 2
    import gc
    gc.collect()
    try:
        os.remove(old)
    except PermissionError:
        pass  # Windows may still hold the temp DB open; harmless


def test_user_voice_filter_keeps_only_remote():
    def tone(f, sec):
        t = np.arange(int(sec * SR)) / SR
        return (0.1 * np.sin(2 * np.pi * f * t)).astype(np.float32)

    def embed(w):                                   # fake: 180 Hz = user (axis 0), 400 Hz = remote (axis 1)
        k = int(np.argmax(np.abs(np.fft.rfft(w))))
        hz = k * SR / len(w)
        v = np.zeros(4); v[0 if hz < 300 else 1] = 1.0
        return v
    user_ref = np.array([1.0, 0, 0, 0])
    flt = UserVoiceFilter(embed, user_ref, drop_above=0.5, slice_s=1.0)
    mixed = np.concatenate([tone(180, 3), tone(400, 3), tone(180, 2), tone(400, 2)])
    out = np.concatenate([flt.process(mixed[i:i + 1600]) for i in range(0, len(mixed), 1600)])
    assert abs(flt.dropped_s - 5.0) < 0.01          # all user slices removed
    assert abs(len(out) / SR - 5.0) < 0.01          # only remote audio kept
    # end to end: analyzer only ever scores remote audio
    seen = []
    an = LiveAnalyzer(lambda w: seen.append(np.abs(np.fft.rfft(w[:SR])).argmax()) or 0.1,
                      gate=GateChain(EnergyVAD(), UserVoiceFilter(embed, user_ref, 0.5)), min_s=2.0)
    ev = feed(an, mixed)
    assert ev and seen and all(k >= 300 for k in seen)   # every scored window is the 400 Hz remote voice
    assert ev[-1]["user_dropped_s"] >= 4.0


def test_pick_caller():
    t, m = 0.30, 0.10
    # clear case: A is the user -> caller is B (in either order)
    assert pick_caller({"A": 0.7, "B": 0.05}, t, m, "A") == ("B", "A", "ok")
    assert pick_caller({"A": 0.05, "B": 0.7}, t, m, "A") == ("A", "B", "ok")
    # nobody sounds like the user -> manual
    assert pick_caller({"A": 0.1, "B": 0.0}, t, m, "A")[2] == "no_match"
    # both sound like the user (one person split in two) -> manual
    assert pick_caller({"A": 0.8, "B": 0.6}, t, m, "A")[2] == "both_match"
    # single voice
    assert pick_caller({"A": 0.9}, t, m, "A")[2] == "single"
    # borderline user match (inside the uncertain zone) still counts as the user's voice
    assert pick_caller({"A": 0.32, "B": 0.0}, t, m, "A") == ("B", "A", "ok")


def test_pause_limiter_keeps_short_pauses_caps_long_ones():
    g = PauseLimiter(max_pause_s=0.6)
    short = np.concatenate([speechlike(1), silence(0.3), speechlike(1)])       # natural pause: kept whole
    assert abs(len(g.process(short)) / SR - 2.3) < 0.05
    g = PauseLimiter(max_pause_s=0.6)
    long_ = np.concatenate([speechlike(2), silence(5), speechlike(2)])          # long gap: capped at ~0.6 s
    assert abs(len(g.process(long_)) / SR - 4.6) < 0.1
    # a silent stream never fills a scoring window
    calls = []
    an = LiveAnalyzer(lambda w: calls.append(1) or 0.9, gate=PauseLimiter(), min_s=2.0)
    feed(an, silence(20))
    assert calls == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
