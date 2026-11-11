"""Rolling-window live analysis of ONE audio stream (the remote party's).

Feed audio chunks with push(); get events back. Pure numpy: scoring functions are injected,
so this file has no torch / streamlit dependency and can be unit-tested with fake scorers.

    score_fn(window: np.ndarray) -> spoof probability (0..1)
    ident_fn(window: np.ndarray) -> similarity to a reference voice, or None (optional)

New in this version (all optional, defaults keep the old behaviour):
  * gate=EnergyVAD(): only SPEECH samples enter the scoring buffer. On a remote-only stream the
    other side is silent whenever you talk; scoring silence/noise gives meaningless spoof scores.
  * smooth=k: median of the last k window scores before thresholding (kills single-window spikes).
  * If pushes arrive faster than the model can score, the analyzer scores the newest window once
    and skips the missed hops instead of re-scoring the same audio (real-time catch-up).
  * SessionVoiceTracker: enrols the remote voice from its first seconds of speech, then flags a
    mid-call change of speaker (voice swap) without needing any saved voice.
"""
import time
from collections import deque

import numpy as np

SR = 16000


def risk_level(p, low=0.30, high=0.70):
    return "Low" if p < low else "High" if p > high else "Uncertain"


class EnergyVAD:
    """Dependency-free energy VAD with an adaptive noise floor and a hangover.

    Good enough to drop silence on a clean remote stream. On noisy phone audio replace it with a
    neural VAD (e.g. Silero) exposing the same .process(samples) -> speech_samples interface.
    """

    def __init__(self, sr=SR, frame_ms=20, hang_ms=300, margin_db=10.0, floor_db=-55.0):
        self.fr = int(sr * frame_ms / 1000)
        self.hang_frames = max(1, int(hang_ms / frame_ms))
        self.margin_db, self.floor_db = margin_db, floor_db
        self.noise_db = floor_db
        self._rest = np.zeros(0, dtype=np.float32)
        self._hang = 0

    def process(self, samples):
        x = np.concatenate([self._rest, np.asarray(samples, dtype=np.float32).ravel()])
        n = len(x) // self.fr
        self._rest = x[n * self.fr:]
        out = []
        for i in range(n):
            f = x[i * self.fr:(i + 1) * self.fr]
            db = 10.0 * np.log10(float(np.mean(f * f)) + 1e-10)
            is_speech = db > max(self.noise_db + self.margin_db, self.floor_db)
            if is_speech:
                self._hang = self.hang_frames
            else:  # track the noise floor on non-speech frames only
                self.noise_db = db if db < self.noise_db else self.noise_db + 0.05 * (db - self.noise_db)
            if is_speech or self._hang > 0:
                if not is_speech:
                    self._hang -= 1
                out.append(f)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class PauseLimiter:
    """Keeps natural pauses but caps every silent stretch at `max_pause_s`.

    Why not EnergyVAD? Joining only the speech removes all pauses. AASIST (trained on ASVspoof
    2019) scored the same real recording 0.48 raw but 0.86 speech-only, so it reacts to pauses.
    This gate still stops long silences (e.g. the remote side is quiet while you talk) from
    filling the scoring window, but leaves the audio sounding like normal speech.
    Same interface as EnergyVAD: .process(samples) -> samples to keep.
    """

    def __init__(self, sr=SR, frame_ms=20, max_pause_s=0.6, margin_db=10.0, floor_db=-55.0):
        self.fr = int(sr * frame_ms / 1000)
        self.max_frames = max(1, int(max_pause_s * 1000 / frame_ms))
        self.margin_db, self.floor_db = margin_db, floor_db
        self.noise_db = floor_db
        self._rest = np.zeros(0, dtype=np.float32)
        self._run = 0                      # consecutive non-speech frames kept so far

    def process(self, samples):
        x = np.concatenate([self._rest, np.asarray(samples, dtype=np.float32).ravel()])
        n = len(x) // self.fr
        self._rest = x[n * self.fr:]
        out = []
        for i in range(n):
            f = x[i * self.fr:(i + 1) * self.fr]
            db = 10.0 * np.log10(float(np.mean(f * f)) + 1e-10)
            if db > max(self.noise_db + self.margin_db, self.floor_db):
                self._run = 0
                out.append(f)
            else:
                self.noise_db = db if db < self.noise_db else self.noise_db + 0.05 * (db - self.noise_db)
                self._run += 1
                if self._run <= self.max_frames:
                    out.append(f)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class UserVoiceFilter:
    """Removes the LOCAL user's own voice from a mixed stream, keeping only the other party.
    Audio is cut into `slice_s` slices; each slice is embedded and compared with the user's
    enrolled voiceprint. Slices that sound like the user (cosine >= drop_above) are dropped.
    replace=False (old behaviour): dropped slices are deleted, so the kept audio is stitched
        together. Both detectors score stitched, pause-free audio as fake (silence_experiment.py).
    replace=True: a dropped slice is replaced by faint noise of the same length, so the timing
        and pauses of the other party's speech survive. Put a PauseLimiter AFTER this filter so
        a long user turn does not fill the scoring window with silence.
    hysteresis: once dropping, keep dropping until the similarity falls below
        drop_above - hysteresis (stops a turn from flickering on and off).
    silence_db: slices quieter than this (mean power, dB) are passed through untouched without
        being embedded (None = embed everything, the old behaviour).
    embed_fn(slice) -> 1-D vector;  user_ref -> the user's enrolled embedding (same model!).
    """
    def __init__(self, embed_fn, user_ref, drop_above, slice_s=1.0, sr=SR,
                 replace=False, hysteresis=0.0, silence_db=None, seed=0):
        r = np.asarray(user_ref, dtype=np.float64)
        self.embed_fn, self.ref = embed_fn, r / (np.linalg.norm(r) + 1e-12)
        self.drop_above, self.n = drop_above, int(slice_s * sr)
        self.replace, self.hyst, self.silence_db = replace, hysteresis, silence_db
        self._rng = np.random.default_rng(seed)
        self._floor_db = -62.0
        self._dropping = False
        self._rest = np.zeros(0, dtype=np.float32)
        self.kept_samples = self.dropped_samples = 0
        self.last_sim = None
    @property
    def dropped_s(self):
        return self.dropped_samples / SR
    def _quiet(self, piece):
        amp = 10.0 ** (self._floor_db / 20.0)
        return (amp * self._rng.standard_normal(len(piece))).astype(np.float32)
    def process(self, samples):
        x = np.concatenate([self._rest, np.asarray(samples, dtype=np.float32).ravel()])
        out = []
        while len(x) >= self.n:
            piece, x = x[:self.n], x[self.n:]
            if self.silence_db is not None:
                db = 10.0 * np.log10(float(np.mean(piece * piece)) + 1e-10)
                if db < self.silence_db:          # a pause: nothing to identify, keep it as it is
                    self._floor_db = float(np.clip(0.9 * self._floor_db + 0.1 * db, -75.0, -50.0))
                    self.kept_samples += len(piece)
                    out.append(piece)
                    continue
            e = np.asarray(self.embed_fn(piece), dtype=np.float64)
            self.last_sim = float(self.ref @ (e / (np.linalg.norm(e) + 1e-12)))
            limit = self.drop_above - (self.hyst if self._dropping else 0.0)
            self._dropping = self.last_sim >= limit
            if self._dropping:
                self.dropped_samples += len(piece)
                if self.replace:
                    out.append(self._quiet(piece))
            else:
                self.kept_samples += len(piece)
                out.append(piece)
        self._rest = x
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class GateChain:
    """Run several gates in order: GateChain(EnergyVAD(), UserVoiceFilter(...))."""

    def __init__(self, *gates):
        self.gates = gates

    def process(self, x):
        for g in self.gates:
            x = g.process(x)
        return x

    @property
    def dropped_s(self):
        return sum(getattr(g, "dropped_s", 0.0) for g in self.gates)


class SessionVoiceTracker:
    """Callable ident_fn: similarity of each window to the voice heard at the START of the call.

    embed_fn(window) -> 1-D vector (e.g. speaker.embed(...).numpy()). The first `enrol_s` seconds
    of speech build the reference (returns None until then). After that, `changed` becomes True
    when `consec` consecutive windows fall below `change_below` (cosine). The threshold is
    model-specific: use the 'uncertain' lower bound you calibrated with calibrate_speaker.py.
    """

    def __init__(self, embed_fn, enrol_s=8.0, change_below=0.15, consec=3, sr=SR):
        self.embed_fn, self.enrol_n = embed_fn, int(enrol_s * sr)
        self.change_below, self.consec = change_below, consec
        self._seen, self._embs, self.ref = 0, [], None
        self.low_run, self.changed = 0, False

    def __call__(self, window):
        if self.ref is None:
            self._embs.append(np.asarray(self.embed_fn(window), dtype=np.float64))
            self._seen += len(window)
            if self._seen >= self.enrol_n:
                r = np.mean(self._embs, axis=0)
                self.ref = r / (np.linalg.norm(r) + 1e-12)
            return None
        e = np.asarray(self.embed_fn(window), dtype=np.float64)
        sim = float(self.ref @ (e / (np.linalg.norm(e) + 1e-12)))
        self.low_run = self.low_run + 1 if sim < self.change_below else 0
        if self.low_run >= self.consec:
            self.changed = True
        return sim


class LiveAnalyzer:
    def __init__(self, score_fn, ident_fn=None, sr=SR, win_s=4.0, hop_s=1.0, min_s=2.0,
        low=0.30, high=0.70, consec=3, gate=None, smooth=1, check_quality=False):
        self.score_fn, self.ident_fn, self.gate = score_fn, ident_fn, gate
        self.sr, self.win, self.hop, self.min = sr, int(win_s * sr), int(hop_s * sr), int(min_s * sr)
        self.low, self.high, self.consec = low, high, consec
        self.buf = np.zeros(0, dtype=np.float32)   # last `win` samples that were scored-eligible
        self.total = 0           # raw samples received (wall-clock position in the call)
        self.speech_total = 0    # samples that passed the gate
        self.next_eval = self.min
        self.streak = 0
        self.alert_active = False
        self.recent = deque(maxlen=max(1, smooth))
        self.history = []
        self.check_quality = check_quality   # True: narrowband audio never raises an alert (see quality.py)
        

    def push(self, samples):
        """Add mono float32 samples at self.sr. Returns the list of new events (0 or 1)."""
        samples = np.asarray(samples, dtype=np.float32).ravel()
        self.total += len(samples)
        if self.gate is not None:
            samples = self.gate.process(samples)
        self.buf = np.concatenate([self.buf, samples])[-self.win:]
        self.speech_total += len(samples)
        if self.speech_total < self.next_eval:
            return []
        ev = self._evaluate()
        while self.next_eval <= self.speech_total:      # skip hops we were too slow to score
            self.next_eval += self.hop
        return [ev]

    def _evaluate(self):
        window = self.buf
        t0 = time.perf_counter()
        raw = float(self.score_fn(window))
        sim = self.ident_fn(window) if self.ident_fn else None
        sim = None if sim is None else float(sim)
        latency_ms = (time.perf_counter() - t0) * 1000
        self.recent.append(raw)
        p = float(np.median(self.recent))
        unreliable = False
        if self.check_quality:
            import quality
            unreliable = bool(quality.check(window, self.sr)["narrowband"])
        self.streak = self.streak + 1 if (p > self.high and not unreliable) else 0
        new_alert = False
        if self.streak >= self.consec and not self.alert_active:
            self.alert_active = new_alert = True
        elif self.streak == 0:
            self.alert_active = False
        ev = {"t": round(self.total / self.sr, 1), "speech_s": round(self.speech_total / self.sr, 1),
              "user_dropped_s": round(float(getattr(self.gate, "dropped_s", 0.0)), 1),
              "p_spoof": p, "p_raw": raw, "similarity": sim,
              "speaker_change": bool(getattr(self.ident_fn, "changed", False)),
              "risk": "High" if self.alert_active else ("Uncertain" if unreliable else risk_level(p, self.low, self.high)),
              "unreliable": unreliable, "new_alert": new_alert, "alert_active": self.alert_active, "latency_ms": latency_ms}
        self.history.append(ev)
        return ev

    def summary(self):
        if not self.history:
            return None
        ps = [e["p_spoof"] for e in self.history]
        first_alert = next((e["t"] for e in self.history if e["new_alert"]), None)
        changed = any(e["speaker_change"] for e in self.history)
        mean = float(np.mean(ps))
        unrel = float(np.mean([e.get("unreliable", False) for e in self.history]))
        
        return {
            "duration": self.total / self.sr,
            "speech_s": self.speech_total / self.sr,
            "peak_spoof": max(ps),
            "mean_spoof": mean,
            "first_alert_t": first_alert,
            "speaker_change": changed,
            "unreliable_frac": unrel,
            "mean_latency_ms": float(np.mean(
                [e["latency_ms"] for e in self.history]
            )),
            "risk": (
                "High" if (first_alert is not None or changed)
                else "Uncertain" if unrel > 0
                else risk_level(mean, self.low, self.high)
            )
        }
