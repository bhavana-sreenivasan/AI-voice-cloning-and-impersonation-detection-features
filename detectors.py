"""One entry point for both anti-spoofing models.

    from detectors import load_model, spoof_prob_window
    model, device = load_model("ssl")          # or "aasist"; default from env SPOOF_MODEL
    p = spoof_prob_window(model, device, wav)  # wav: float32 mono 16 kHz, any length <= ~4 s

AASIST returns (hidden, logits); SSL-AASIST returns logits only. `forward_logits` hides that.
Index 0 = spoof, index 1 = bonafide for both (same convention as the original code).

This module imports no model code at import time, so pipeline.py works with either model
installed alone (SSL-AASIST needs the fairseq environment, AASIST does not).
"""
import os

import numpy as np
import torch

SR = 16000
MAX_LEN = 64600  # ~4 s, the input length both models were trained on


def pad_to(x, n=MAX_LEN):
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def default_name():
    return os.environ.get("SPOOF_MODEL", "aasist").lower()


def load_model(name=None, device=None):
    name = (name or default_name()).lower()
    if name == "ssl":
        from spoof_ssl import load_model as _load      # needs fairseq + xlsr2_300m.pt + LA_model.pth
    elif name == "aasist":
        from spoof import load_model as _load
    else:
        raise ValueError("model must be 'aasist' or 'ssl', got " + name)
    return _load(device)


def forward_logits(model, batch):
    """-> (n, 2) logits for either model."""
    out = model(batch)
    return out[1] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def spoof_prob_window(model, device, wav):
    """Spoof probability (0..1) for one window; short windows are tiled up to 4 s."""
    x = torch.tensor(pad_to(np.asarray(wav, dtype=np.float32))[None], dtype=torch.float32).to(device)
    return float(torch.softmax(forward_logits(model, x), dim=1)[0, 0])
