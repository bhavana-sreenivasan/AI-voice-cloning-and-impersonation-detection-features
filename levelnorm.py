"""levelnorm.py - steady speech level before the spoof detector.

Why: AASIST's score moved with loudness alone (same audio: x0.5 -> 0.33, x2 -> 0.67).
A call has a different volume from one recording to the next (phone, WhatsApp, speaker-phone),
so without this the alert depends on how loud the line is, not on whether the voice is real.

How: keeps the last `history_s` seconds of 20 ms frame levels and takes their 95th percentile
as "the speech level" (loud frames = speech; this shifts one-for-one with gain, so it does not
depend on a silence threshold). The audio is scaled so that level hits `target_db`. The gain is
smoothed, capped at +-`max_gain_db`, the audio is hard-limited to +-1, and the length never
changes, so it can sit anywhere in a GateChain (same interface as EnergyVAD / PauseLimiter):

    GateChain(PauseLimiter(), UserVoiceFilter(...), LevelNormalizer())

If there is no clear speech yet (the loud frames are not at least `margin_db` above the quiet
ones, or everything is below `floor_db`) the gain is left unchanged, so line noise or silence is
never boosted up to speech level.  It does NOT remove noise and does NOT touch pauses.

CHOOSING target_db: the detectors were trained on ASVspoof LA audio, so match its level. Run
    from levelnorm import p95_db
    # median of p95_db(wav) over ~100 bonafide LA clips  ->  use that as target_db
"""
from collections import deque

import numpy as np

SR = 16000


def frame_db(x, fr):
    n = len(x) // fr
    if n == 0:
        return np.zeros(0)
    f = np.asarray(x[: n * fr], dtype=np.float32).reshape(n, fr)
    return 10.0 * np.log10(np.mean(f * f, axis=1) + 1e-10)


def p95_db(x, sr=SR, frame_ms=20):
    """95th-percentile frame level (dB) of a waveform: use it to pick target_db."""
    d = frame_db(np.asarray(x, dtype=np.float32).ravel(), int(sr * frame_ms / 1000))
    return float(np.percentile(d, 95)) if d.size else float("-inf")


class LevelNormalizer:
    def __init__(self, sr=SR, frame_ms=20, target_db=-20.0, max_gain_db=18.0, margin_db=10.0,
                 floor_db=-60.0, history_s=8.0, min_frames=25, gain_alpha=0.3):
        self.fr = int(sr * frame_ms / 1000)
        self.target_db, self.max_gain_db = target_db, max_gain_db
        self.margin_db, self.floor_db = margin_db, floor_db
        self.min_frames, self.gain_alpha = min_frames, gain_alpha
        self.hist = deque(maxlen=max(1, int(history_s * 1000 / frame_ms)))
        self.gain_db = 0.0            # gain currently applied (dB)
        self._target_gain_db = 0.0
        self._locked = False          # becomes True once clear speech has been heard

    def process(self, samples):
        x = np.asarray(samples, dtype=np.float32).ravel()
        if x.size == 0:
            return x
        self.hist.extend(frame_db(x, self.fr).tolist())
        if len(self.hist) >= self.min_frames:
            h = np.asarray(self.hist)
            loud, quiet = float(np.percentile(h, 95)), float(np.percentile(h, 10))
            if loud > self.floor_db and loud - quiet >= self.margin_db:     # clear speech present
                self._target_gain_db = float(np.clip(self.target_db - loud, -self.max_gain_db, self.max_gain_db))
                if not self._locked:                                         # first time: don't creep up from 0 dB
                    self.gain_db, self._locked = self._target_gain_db, True
        new_db = self.gain_db + self.gain_alpha * (self._target_gain_db - self.gain_db)
        g0, g1 = 10.0 ** (self.gain_db / 20.0), 10.0 ** (new_db / 20.0)
        self.gain_db = new_db
        return np.clip(x * np.linspace(g0, g1, x.size, dtype=np.float32), -1.0, 1.0)
