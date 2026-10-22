"""Rolling-window live analysis. Feed audio chunks with push(); get one event per hop.

The scoring functions are injected, so this file has no torch/streamlit dependency:
    score_fn(window: np.ndarray) -> spoof probability (0..1)
    ident_fn(window: np.ndarray) -> similarity to the claimed voice (optional)
"""
import time

import numpy as np

SR = 16000


def risk_level(p, low=0.30, high=0.70):
    return "Low" if p < low else "High" if p > high else "Uncertain"


class LiveAnalyzer:
    def __init__(self, score_fn, ident_fn=None, sr=SR, win_s=4.0, hop_s=1.0, min_s=2.0,
                 low=0.30, high=0.70, consec=3):
        self.score_fn, self.ident_fn = score_fn, ident_fn
        self.sr, self.win, self.hop, self.min = sr, int(win_s * sr), int(hop_s * sr), int(min_s * sr)
        self.low, self.high, self.consec = low, high, consec
        self.buf = np.zeros(0, dtype=np.float32)
        self.total = 0           # samples received
        self.next_eval = self.min
        self.streak = 0
        self.alert_active = False
        self.history = []

    def push(self, samples):
        """Add mono float32 samples at self.sr. Returns the list of new events."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        self.buf = np.concatenate([self.buf, samples])[-self.win:]
        self.total += len(samples)
        events = []
        while self.total >= self.next_eval:
            events.append(self._evaluate(self.next_eval))
            self.next_eval += self.hop
        return events

    def _evaluate(self, end_sample):
        # Window ends at the newest buffered audio (chunks are processed as they arrive).
        window = self.buf
        t0 = time.perf_counter()
        p = float(self.score_fn(window))
        sim = float(self.ident_fn(window)) if self.ident_fn else None
        latency_ms = (time.perf_counter() - t0) * 1000
        self.streak = self.streak + 1 if p > self.high else 0
        new_alert = False
        if self.streak >= self.consec and not self.alert_active:
            self.alert_active = new_alert = True
        elif self.streak == 0:
            self.alert_active = False
        ev = {"t": round(end_sample / self.sr, 1), "p_spoof": p, "similarity": sim,
              "risk": "High" if self.alert_active else risk_level(p, self.low, self.high),
              "new_alert": new_alert, "alert_active": self.alert_active, "latency_ms": latency_ms}
        self.history.append(ev)
        return ev

    def summary(self):
        if not self.history:
            return None
        ps = [e["p_spoof"] for e in self.history]
        first_alert = next((e["t"] for e in self.history if e["new_alert"]), None)
        return {"duration": self.total / self.sr, "peak_spoof": max(ps), "mean_spoof": float(np.mean(ps)),
                "first_alert_t": first_alert, "mean_latency_ms": float(np.mean([e["latency_ms"] for e in self.history])),
                "risk": "High" if first_alert is not None else risk_level(float(np.mean(ps)), self.low, self.high)}
