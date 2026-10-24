"""Split a call recording into speakers so each voice is analysed on its own."""
import io

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering

import speaker  # reuses your ECAPA embeddings

SR = 16000
WIN, HOP = int(1.5 * SR), int(0.75 * SR)
SAME_SPEAKER_COS = 0.55   # centroids more similar than this => treat as ONE speaker (tune it)


def _speech_only(x, top_db=30):
    spans = librosa.effects.split(x, top_db=top_db)
    return np.concatenate([x[s:e] for s, e in spans]) if len(spans) else x


def _by_channel(y):
    """Stereo call recorders often put each side on its own channel: use that if so."""
    if y.ndim != 2 or y.shape[0] != 2:
        return None
    a, b = y
    n = min(len(a), len(b))
    if a[:n].std() < 1e-4 or b[:n].std() < 1e-4 or abs(np.corrcoef(a[:n], b[:n])[0, 1]) > 0.8:
        return None  # silent channel or dual-mono
    return [_speech_only(a), _speech_only(b)]


def _by_clustering(wav):
    spans = librosa.effects.split(wav, top_db=30)
    wins = [(st, st + WIN) for s, e in spans for st in range(s, e - WIN + 1, HOP)]
    if len(wins) < 4:
        return [_speech_only(wav)]
    embs = np.stack([speaker.embed(torch.tensor(wav[a:b])).numpy() for a, b in wins])
    labels = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(embs)
    if min((labels == 0).sum(), (labels == 1).sum()) < 2:
        return [_speech_only(wav)]
    c = [embs[labels == k].mean(0) for k in (0, 1)]
    c = [v / np.linalg.norm(v) for v in c]
    if float(c[0] @ c[1]) > SAME_SPEAKER_COS:
        return [_speech_only(wav)]
    # give each window's middle HOP samples to its cluster (avoids duplicating overlap)
    mask = np.full(len(wav), -1)
    for (a, b), k in zip(wins, labels):
        mid = (a + b) // 2
        mask[mid - HOP // 2: mid + HOP // 2] = k
    return [wav[mask == k] for k in (0, 1)]


def split(path):
    """-> list of float32 waveforms, longest speaker first."""
    y, _ = librosa.load(path, sr=SR, mono=False)
    parts = _by_channel(y)
    if parts is None:
        parts = _by_clustering(y if y.ndim == 1 else y.mean(axis=0))
    return sorted(parts, key=len, reverse=True)


def to_wav_bytes(x):
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV")
    return buf.getvalue()
