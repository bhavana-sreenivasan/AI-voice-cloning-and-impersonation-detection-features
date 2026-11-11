"""quality.py - is this audio wide enough for the spoof detectors to be trusted?

Why: both detectors (AASIST, SSL-AASIST) were trained on wideband 16 kHz studio-style audio.
On phone-quality audio (WhatsApp / GSM / VoIP: 8 kHz, nothing above ~4 kHz, codec-compressed)
they give scores that depend on loudness and pauses, not on whether the voice is real: a real
WhatsApp call scored anywhere from 0.05 to 0.99.  So when the audio is narrowband the spoof score
must not raise (or clear) an alert on its own.

How: average power spectrum of the speech frames; hf_db = energy in the 4.5-7.5 kHz band relative to
the 1-3.4 kHz band, in dB.  Comparing with the mid band (not the total) cancels how bright the voice is.
Measured: 200 real ASVspoof LA clips span -34 dB (darkest) to about -10 dB (median -15);
simulated 8 kHz audio gives -49 to -59 dB; a real WhatsApp call gave -71 dB.  The threshold sits in
the gap (-42 dB), about 8 dB from the darkest wideband clip and 7 dB from the brightest narrowband case.
narrowband = hf_db < NARROWBAND_DB.  numpy only.

Check the threshold on your own data (should show wideband LA clips far above it):
    python quality.py "C:\\...\\data\\LA\\ASVspoof2019_LA_eval\\flac" --n 200
    python quality.py rec.aac
"""
import os
import sys

import numpy as np

SR = 16000
HI_BAND = (4500.0, 7500.0)  # empty in 8 kHz audio
MID_BAND = (1000.0, 3400.0)  # present in all speech
NARROWBAND_DB = -42.0       # HI/MID energy ratio below this -> narrowband (see the numbers in the docstring)
_NFFT = 4096


def hf_ratio_db(x, sr=SR):
    """Energy in HI_BAND relative to MID_BAND (dB) over the speech frames.
    None if the signal is silent / too short to say."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) < _NFFT // 2:
        return None
    if len(x) < _NFFT:
        x = np.pad(x, (0, _NFFT - len(x)))
    n = len(x) // _NFFT
    fr = x[: n * _NFFT].reshape(n, _NFFT)
    lev = 10.0 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)
    if float(lev.max()) < -70.0:                       # silence
        return None
    active = lev > (np.percentile(lev, 95) - 30.0)     # speech frames only
    spec = (np.abs(np.fft.rfft(fr[active] * np.hanning(_NFFT), axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(_NFFT, 1.0 / sr)
    band = lambda lo, hi: float(spec[(f >= lo) & (f <= hi)].sum())   # noqa: E731
    mid = band(*MID_BAND)
    if mid <= 0.0:
        return None
    return float(10.0 * np.log10(band(*HI_BAND) / mid + 1e-15))


def check(x, sr=SR, threshold_db=NARROWBAND_DB):
    """-> {'hf_db': float | None, 'narrowband': bool}.  Unknown (silence) is NOT flagged."""
    hf = hf_ratio_db(x, sr)
    return {"hf_db": hf, "narrowband": hf is not None and hf < threshold_db}


# --------------------------------------------------------------------------- CLI
def _load(path):
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), SR)
            x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        return x
    except Exception:                                   # aac / m4a / mp3: use the project loader
        from capture import load_audio_16k
        return load_audio_16k(path)


def _collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out += [os.path.join(root, f) for f in files if f.lower().endswith((".flac", ".wav"))]
        else:
            out.append(p)
    return sorted(out)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="audio files and/or folders of .flac/.wav")
    ap.add_argument("--n", type=int, default=0, help="max files (0 = all)")
    ap.add_argument("--threshold", type=float, default=NARROWBAND_DB)
    a = ap.parse_args(argv)
    files = _collect(a.paths)
    if a.n:
        import random
        random.Random(0).shuffle(files)
        files = files[: a.n]
    vals = []
    for f in files:
        r = check(_load(f), threshold_db=a.threshold)
        if r["hf_db"] is not None:
            vals.append(r["hf_db"])
        if len(files) <= 10:
            hf = "n/a" if r["hf_db"] is None else f"{r['hf_db']:.1f} dB"
            print(f"{os.path.basename(f)}: 4.5-7.5 kHz vs 1-3.4 kHz = {hf}  ->  "
                  f"{'NARROWBAND (spoof score unreliable)' if r['narrowband'] else 'wideband'}")
    if len(files) > 10 and vals:
        v = np.array(vals)
        print(f"{len(v)} files: energy 4.5-7.5 kHz relative to 1-3.4 kHz (dB)")
        for q in (0, 1, 5, 25, 50):
            print(f"  p{q:02d} = {np.percentile(v, q):7.1f}")
        print(f"narrowband (< {a.threshold:.0f} dB): {int((v < a.threshold).sum())} of {len(v)}")
        print("Wideband clips should all sit well ABOVE the threshold (keep >5 dB of margin at p00); phone audio well below it.")


if __name__ == "__main__":
    main(sys.argv[1:])