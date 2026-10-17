"""Spoof detector using pretrained AASIST (clovaai/aasist).

Setup (once):
    git clone https://github.com/clovaai/aasist
    # pretrained weights are in aasist/models/weights/AASIST.pth
Run this project from the folder that contains the `aasist/` folder.
"""
import sys
import json
import numpy as np
import torch
import librosa

sys.path.append("aasist")
from models.AASIST import Model  # noqa: E402

SR = 16000
MAX_LEN = 64600  # ~4 sec, input length AASIST was trained on


def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open("aasist/config/AASIST.conf") as f:
        cfg = json.load(f)
    model = Model(cfg["model_config"]).to(device)
    state = torch.load("aasist/models/weights/AASIST.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _pad(x, n=MAX_LEN):
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def _chunks(wav):
    """Split waveform into 4s chunks (last one padded by repeating)."""
    if len(wav) <= MAX_LEN:
        return [_pad(wav)]
    out = [wav[i:i + MAX_LEN] for i in range(0, len(wav) - MAX_LEN + 1, MAX_LEN)]
    rest = len(wav) % MAX_LEN
    if rest > SR:  # keep leftover only if longer than 1s
        out.append(_pad(wav[-rest:]))
    return out


@torch.no_grad()
def spoof_scores(path, model, device):
    """Returns (bonafide_logit_score, spoof_probability) averaged over chunks.

    Higher bonafide_logit_score = more likely real (use this for EER).
    """
    wav, _ = librosa.load(path, sr=SR, mono=True)
    if len(wav) == 0:
        raise ValueError(f"Empty audio: {path}")
    batch = torch.tensor(np.stack(_chunks(wav)), dtype=torch.float32).to(device)
    _, out = model(batch)  # out shape: (n_chunks, 2); index 1 = bonafide
    logit_score = out[:, 1].mean().item()
    spoof_prob = torch.softmax(out, dim=1)[:, 0].mean().item()
    return logit_score, spoof_prob


if __name__ == "__main__":
    model, device = load_model()
    s, p = spoof_scores(sys.argv[1], model, device)
    print(f"bonafide score: {s:.3f} | spoof probability: {p:.2%}")
