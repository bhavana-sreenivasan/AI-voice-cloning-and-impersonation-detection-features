"""aasist_io.py - load AASIST with the pretrained OR a fine-tuned checkpoint.

    import aasist_io, detectors
    model, device = aasist_io.load()                       # pretrained aasist/models/weights/AASIST.pth
    model, device = aasist_io.load("finetune_out/best.pth")  # your fine-tuned weights
    p = detectors.spoof_prob_window(model, device, wav)    # same call as everywhere else

Run from anywhere: paths are relative to this file, not to the working folder.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "aasist", "models", "weights", "AASIST.pth")


def build():
    """AASIST with the architecture from aasist/config/AASIST.conf (random weights)."""
    path = os.path.join(HERE, "aasist")
    if path not in sys.path:
        sys.path.insert(0, path)
    from models.AASIST import Model
    with open(os.path.join(path, "config", "AASIST.conf")) as f:
        cfg = json.load(f)
    return Model(cfg["model_config"])


def load(checkpoint=None, device=None, train=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build()
    state = torch.load(checkpoint or DEFAULT_WEIGHTS, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.train(train)
    return model, device