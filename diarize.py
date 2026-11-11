"""Split a call recording into speakers so each voice is analysed on its own. diarize.py"""
import io, os
import audio_io

def _load_any(path):
    with open(path, "rb") as f:
        data, _ = audio_io.to_wav(f.read(), os.path.splitext(path)[1] or ".wav")
    return librosa.load(io.BytesIO(data), sr=SR, mono=False)

# in split():  y, _ = _load_any(path)     # instead of librosa.load(path, sr=SR, mono=False)
import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering

import speaker  # reuses your ECAPA embeddings

SR = 16000
WIN, HOP = int(1.5 * SR), int(0.75 * SR)
SAME_SPEAKER_COS = 0.55   # centroids more similar than this => treat as ONE speaker (tune it)
MERGE_GAP_S = 0.5         # join speech spans closer than this: choppy / noise-gated phone audio is cut into
                          # many tiny spans, which leaves too few 1.5 s windows to cluster (0 = off)


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

def _merge_close(spans, gap):
    """Join speech spans separated by less than `gap` samples."""
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def _by_clustering(wav):
    spans = librosa.effects.split(wav, top_db=30)
    n_raw = len(spans)
    if MERGE_GAP_S > 0 and n_raw:
        spans = _merge_close(spans, int(MERGE_GAP_S * SR))
        print(f"Speech spans merged (gaps < {MERGE_GAP_S}s): {n_raw} -> {len(spans)}")

    wins = [
        (st, st + WIN)
        for s, e in spans
        for st in range(s, e - WIN + 1, HOP)
    ]

    print(f"Speech spans: {len(spans)}")
    print(f"Generated windows: {len(wins)}")
    print("Window durations:", [(b-a)/SR for a, b in wins])

    if len(wins) < 4:
        return [_speech_only(wav)]

    embs = np.stack([
        speaker.embed(torch.as_tensor(wav[a:b], dtype=torch.float32))
        .detach().cpu().numpy()
        for a, b in wins
    ])

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - SAME_SPEAKER_COS,   # = 0.45, ties the cut-off to the constant above
        metric="cosine",
        linkage="average"
    )

    labels = clustering.fit_predict(embs)
    unique = np.unique(labels)

    print(f"Detected clusters: {len(unique)}")

    if len(unique) < 2:
        return [_speech_only(wav)]

    mask = np.full(len(wav), -1, dtype=int)

    for (a, b), k in zip(wins, labels):
        mid = (a + b) // 2
        start = mid - HOP // 2
        end = mid + HOP // 2
        mask[start:end] = k

    parts = [
        wav[mask == k]
        for k in unique
        if np.sum(mask == k) > 0
    ]

    return parts



    


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