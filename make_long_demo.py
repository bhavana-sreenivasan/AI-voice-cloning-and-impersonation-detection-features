"""Build long_demo.wav = 3 real clips followed by 3 fake clips (mid-call swap demo).

Run from the project folder:
    python make_long_demo.py
"""
import os
import numpy as np
import soundfile as sf

ROOT = os.path.join("data", "LA")
PROTO = os.path.join(ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
AUDIO = os.path.join(ROOT, "ASVspoof2019_LA_eval", "flac")
N_EACH = 3

real, fake = [], []
with open(PROTO) as f:
    for line in f:
        p = line.split()
        if len(p) < 5:
            continue
        (real if p[4] == "bonafide" else fake).append(p[1])

parts = [sf.read(os.path.join(AUDIO, u + ".flac"))[0] for u in real[:N_EACH] + fake[:N_EACH]]
audio = np.concatenate(parts)
sf.write("long_demo.wav", audio, 16000)

print(f"Saved long_demo.wav ({len(audio) / 16000:.1f} s)")
print("Real:", real[:N_EACH])
print("Fake:", fake[:N_EACH])