"""
speaker.py - speaker identity verification.

Default backend: ECAPA2 (Thienpondt & Demuynck, ASRU 2023; HuggingFace `Jenthe/ECAPA2`,
TorchScript, 192-d embedding, 16 kHz input).  NOTE: the ECAPA2 weights are licensed
CC-BY-NC-4.0 (non-commercial). Check this before any commercial deployment.

Fallback backend (for A/B comparison and for the report): the old SpeechBrain ECAPA-TDNN.
Select with the environment variable  SPEAKER_BACKEND=ecapa2 | speechbrain   (default ecapa2).

Public API is unchanged from the ECAPA-TDNN version:
    ref = speaker.enroll(["alice_1.wav", "alice_2.wav"])
    res = speaker.verify_clip("incoming.wav", ref)
    speaker.embed(wav_tensor) -> L2-normalised 1-D tensor
    speaker.BACKEND, speaker.EMB_MODEL  -> which model produced an embedding (store this with it!)

IMPORTANT: embeddings from different backends live in different spaces even when both are
192-d. Never compare them. Thresholds below are UNCALIBRATED for ECAPA2: run
calibrate_speaker.py on your own data and set the values (or the env vars) before trusting
match / mismatch decisions.
"""
import os

import librosa
import numpy as np
import torch

SR = 16000
WIN_S, HOP_S = 4.0, 1.0          # same windowing as the spoof pipeline
MIN_S = 1.0                      # shorter inputs are tiled up to this length

BACKEND = os.environ.get("SPEAKER_BACKEND", "ecapa2").lower()
DEVICE = os.environ.get("SPEAKER_DEVICE", "cpu")
if BACKEND not in ("ecapa2", "speechbrain"):
    raise ValueError("SPEAKER_BACKEND must be 'ecapa2' or 'speechbrain', got " + BACKEND)
EMB_MODEL = BACKEND

# (match threshold, uncertain margin) on cosine similarity.
# speechbrain: 0.25 / 0.10 is what the project used so far.
# ecapa2: PLACEHOLDER copied from the old value. Cosine scales differ between models.
#         Override with SPEAKER_MATCH_T / SPEAKER_MARGIN after running calibrate_speaker.py.
_DEFAULTS = {"speechbrain": (0.25, 0.10), "ecapa2": (0.25, 0.10)}
MATCH_THRESHOLD = float(os.environ.get("SPEAKER_MATCH_T", _DEFAULTS[BACKEND][0]))
UNCERTAIN_MARGIN = float(os.environ.get("SPEAKER_MARGIN", _DEFAULTS[BACKEND][1]))

_model = None


def _load_ecapa2():
    from huggingface_hub import hf_hub_download
    last = None
    for fname in ("ecapa2.pt", "model.pt"):          # the file was renamed on the Hub once
        try:
            path = hf_hub_download(repo_id="Jenthe/ECAPA2", filename=fname)
            return torch.jit.load(path, map_location=DEVICE)
        except Exception as e:                       # noqa: BLE001
            last = e
    raise RuntimeError(f"Could not load ECAPA2 from the HuggingFace Hub: {last}")


def _load_speechbrain():
    try:  # SpeechBrain >= 1.0
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
    )


def _get_model():
    global _model
    if _model is None:
        _model = _load_ecapa2() if BACKEND == "ecapa2" else _load_speechbrain()
    return _model


def load_audio(path):
    """Load any audio file as mono 16 kHz float tensor of shape (samples,)."""
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return torch.tensor(wav, dtype=torch.float32)


def embed(wav):
    """L2-normalised 192-d speaker embedding for a 1-D float waveform at 16 kHz."""
    wav = torch.as_tensor(wav, dtype=torch.float32).flatten()
    n = int(MIN_S * SR)
    if len(wav) < n:                                  # very short windows: tile up
        wav = wav.repeat(int(np.ceil(n / max(len(wav), 1))))[:n]
    x = wav.unsqueeze(0).to(DEVICE)                   # (1, samples)
    with torch.no_grad():
        if BACKEND == "ecapa2":
            with torch.jit.optimized_execution(False):   # avoids very slow first calls
                e = _get_model()(x)                       # ECAPA2 sets eval mode itself
        else:
            e = _get_model().encode_batch(x)
    e = e.reshape(-1).float().cpu()
    return torch.nn.functional.normalize(e, dim=0)


def enroll(paths):
    """Reference voiceprint from one or more clips (30 s+ of clean speech is best)."""
    embs = torch.stack([embed(load_audio(p)) for p in paths])
    return torch.nn.functional.normalize(embs.mean(dim=0), dim=0)


def cosine(a, b):
    return float(torch.dot(a, b))


def decide(score, match_t=None, margin=None):
    """Map a cosine score to 'match' / 'uncertain' / 'mismatch'."""
    t = MATCH_THRESHOLD if match_t is None else match_t
    m = UNCERTAIN_MARGIN if margin is None else margin
    if score >= t + m:
        return "match"
    if score <= t - m:
        return "mismatch"
    return "uncertain"


def verify_clip(path, ref):
    """Compare a whole clip plus per-window scores for a timeline.

    Returns dict: score, decision, match, timeline [(start_s, score)], speaker_change
    (True if 3 consecutive windows fall in the mismatch zone = a mid-clip speaker swap).
    """
    wav = load_audio(path)
    score = cosine(embed(wav), ref)

    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    timeline = []
    if len(wav) <= win:
        timeline.append((0.0, score))
    else:
        for start in range(0, len(wav) - win + 1, hop):
            timeline.append((start / SR, cosine(embed(wav[start:start + win]), ref)))

    low = MATCH_THRESHOLD - UNCERTAIN_MARGIN
    run, change = 0, False
    for _, s in timeline:
        run = run + 1 if s <= low else 0
        if run >= 3:
            change = True
            break

    decision = decide(score)
    return {"score": score, "decision": decision, "match": decision == "match",
            "timeline": timeline, "speaker_change": change}
