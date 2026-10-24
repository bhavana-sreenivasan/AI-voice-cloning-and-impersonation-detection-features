"""Decode any input format (aac, m4a, mp3, ...) to WAV with ffmpeg."""
import os
import subprocess
import tempfile


def to_wav(data, ext):
    """-> (wav_bytes, '.wav'). Keeps the original channels so stereo call recordings still split by side."""
    if ext.lower() == ".wav":
        return data, ext
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as src:
        src.write(data)
    out = src.name + ".wav"
    try:
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src.name, "-ar", "16000", out],
                           capture_output=True)
        if r.returncode != 0:
            raise ValueError("ffmpeg could not decode this file: " + r.stderr.decode()[-300:])
        with open(out, "rb") as f:
            return f.read(), ".wav"
    except FileNotFoundError:
        raise ValueError("ffmpeg is not installed or not on PATH.")
    finally:
        for p in (src.name, out):
            if os.path.exists(p):
                os.remove(p)
