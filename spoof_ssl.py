"""SSL-AASIST detector with the same interface as spoof.py."""
import os, sys, glob
from argparse import Namespace
import numpy as np
import torch
import librosa

HERE = os.path.dirname(os.path.abspath(__file__))

def _find_ssl_dir():
    cands = [HERE] + [os.path.join(HERE, d) for d in os.listdir(HERE)]
    for root, _, files in os.walk(HERE):
        if "fairseq" not in root:
            cands.append(root)
    for c in cands:
        p = os.path.join(c, "model.py")
        if os.path.isfile(p) and "class SSLModel" in open(p, errors="ignore").read():
            return c
    raise FileNotFoundError("SSL-AASIST model.py not found under " + HERE)

SSL_DIR = _find_ssl_dir()
sys.path.insert(0, SSL_DIR)
from model import Model  # noqa: E402

SR = 16000
MAX_LEN = 64600

def _weights():
    p = os.path.join(SSL_DIR, "LA_model.pth")
    if os.path.exists(p):
        return p
    c = sorted(glob.glob(os.path.join(SSL_DIR, "*.pth")))
    if not c:
        raise FileNotFoundError("No .pth weights in " + SSL_DIR)
    return c[0]

def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(Namespace(), device).to(device)
    state = torch.load(_weights(), map_location=device)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, device

def _pad(x, n=MAX_LEN):
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]

def _chunks(wav):
    if len(wav) <= MAX_LEN:
        return [_pad(wav)]
    out = [wav[i:i + MAX_LEN] for i in range(0, len(wav) - MAX_LEN + 1, MAX_LEN)]
    rest = len(wav) % MAX_LEN
    if rest > SR:
        out.append(_pad(wav[-rest:]))
    return out

@torch.no_grad()
def spoof_scores(path, model, device):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    if len(wav) == 0:
        raise ValueError("Empty audio: " + path)
    batch = torch.tensor(np.stack(_chunks(wav)), dtype=torch.float32).to(device)
    out = model(batch)  # (n_chunks, 2); index 1 = bonafide
    return out[:, 1].mean().item(), torch.softmax(out, dim=1)[:, 0].mean().item()

if __name__ == "__main__":
    m, d = load_model()
    s, p = spoof_scores(sys.argv[1], m, d)
    print(f"bonafide score: {s:.3f} | spoof probability: {p:.2%}")
