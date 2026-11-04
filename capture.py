"""Audio sources that yield ONLY the remote party's audio as float32 mono 16 kHz chunks.

The cleanest way to analyse "only the opponent" is to never record your own microphone:

  * LoopbackSource   - desktop calls (Zoom/Meet/Teams/WhatsApp Desktop/softphone): records what the
                       computer PLAYS, which is the remote voice only. Route the call app to its own
                       output device (VB-Cable on Windows, BlackHole on macOS) so music or
                       notification sounds are not captured too.
  * ArraySource      - a numpy array / file, optionally paced in real time. Use it to test, demo and
                       evaluate the live pipeline without a real call. Pick the channel that holds
                       the remote side of a stereo call recording.
  * mulaw_decode     - for telephony streams (e.g. Twilio Media Streams `inbound_track` = the remote
                       caller only, 8 kHz mu-law). Decode, then resample_to_16k().

Each source is an iterator of chunks and has .close().
"""
import math
import time

import numpy as np

SR = 16000


def resample_to_16k(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if sr == SR:
        return x
    from scipy.signal import resample_poly
    g = math.gcd(SR, int(sr))
    return resample_poly(x, SR // g, int(sr) // g).astype(np.float32)


def mulaw_decode(data: bytes) -> np.ndarray:
    """G.711 mu-law bytes -> float32 in [-1, 1] (8 kHz telephony). No audioop (removed in Python 3.13)."""
    u = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32)
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    mag = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return (np.where(sign != 0, -mag, mag) / 32768.0).astype(np.float32)


def mulaw_encode(x: np.ndarray) -> bytes:
    """float32 [-1, 1] -> G.711 mu-law bytes (used for tests and for simulating phone audio)."""
    pcm = np.clip(np.asarray(x, dtype=np.float32) * 32768.0, -32635, 32635).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    pcm = np.abs(pcm) + 0x84
    exponent = np.clip(np.floor(np.log2(np.maximum(pcm, 1))).astype(np.int32) - 7, 0, 7)
    mantissa = (pcm >> (exponent + 3)) & 0x0F
    return bytes((~(sign | (exponent << 4) | mantissa)).astype(np.uint8) & 0xFF)


class ArraySource:
    """Chunks of a waveform. realtime=True sleeps so audio arrives at wall-clock speed."""

    def __init__(self, wav, sr=SR, chunk_ms=100, realtime=True):
        self.wav = resample_to_16k(wav, sr)
        self.n = int(SR * chunk_ms / 1000)
        self.realtime, self.chunk_s = realtime, chunk_ms / 1000
        self._stop = False

    @classmethod
    def from_file(cls, path, channel="mix", chunk_ms=100, realtime=True):
        """channel: 0 / 1 = that side of a stereo recording (the remote party), 'mix' = average."""
        import librosa
        y, _ = librosa.load(path, sr=SR, mono=False)
        if y.ndim == 2:
            y = y.mean(axis=0) if channel == "mix" else y[int(channel)]
        return cls(y, SR, chunk_ms, realtime)

    def __iter__(self):
        start = time.perf_counter()
        for i, s in enumerate(range(0, len(self.wav), self.n)):
            if self._stop:
                return
            if self.realtime:
                wait = start + i * self.chunk_s - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            yield self.wav[s:s + self.n]

    def close(self):
        self._stop = True


class LoopbackSource:
    """System-output capture via the `soundcard` package (Windows WASAPI, macOS CoreAudio, Linux Pulse).

    Hardware-dependent: not exercised in the automated tests. On macOS you need a virtual device
    such as BlackHole as the output, since CoreAudio has no native loopback.
    """

    def __init__(self, speaker_name=None, chunk_ms=100, device_sr=48000):
        import soundcard as sc
        spk = sc.get_speaker(speaker_name) if speaker_name else sc.default_speaker()
        self.mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        self.device_sr, self.frames = device_sr, int(device_sr * chunk_ms / 1000)
        self._stop = False

    def __iter__(self):
        with self.mic.recorder(samplerate=self.device_sr, channels=2) as rec:
            while not self._stop:
                data = rec.record(numframes=self.frames)          # (frames, 2)
                yield resample_to_16k(data.mean(axis=1), self.device_sr)

    def close(self):
        self._stop = True
