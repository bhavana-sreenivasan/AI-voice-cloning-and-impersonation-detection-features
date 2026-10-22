"""
speaker.py - speaker identity verification using ECAPA-TDNN (SpeechBrain).

Standalone module: no dependency on spoof.py / pipeline.py.

Install:  pip install speechbrain
Model:    speechbrain/spkrec-ecapa-voxceleb (downloaded on first use, ~80 MB)

Typical use:
    import speaker
    ref = speaker.enroll(["alice_1.wav", "alice_2.wav"])   # once per person
    result = speaker.verify_clip("incoming.wav", ref)      # per clip
    print(result["score"], result["match"], result["timeline"])
"""
import numpy as np
import torch
import librosa

try:  # SpeechBrain >= 1.0
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:  # older versions
    from speechbrain.pretrained import EncoderClassifier

SR = 16000
WIN_S, HOP_S = 4.0, 1.0          # same windowing as your AASIST pipeline
MATCH_THRESHOLD = 0.25           # cosine score; SpeechBrain's recipe uses ~0.25.
                                 # Tune it on your own data (see calibrate note below).
UNCERTAIN_MARGIN = 0.10          # scores within +/- margin of threshold = uncertain

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
    return _model


def load_audio(path):
    """Load any audio file as mono 16 kHz float tensor of shape (samples,).
    Uses librosa, same as pipeline.py, so formats behave identically."""
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return torch.tensor(wav, dtype=torch.float32)


def embed(wav):
    """Return an L2-normalised 192-dim speaker embedding for a waveform tensor."""
    with torch.no_grad():
        e = _get_model().encode_batch(wav.unsqueeze(0)).squeeze()
    return torch.nn.functional.normalize(e, dim=0)


def enroll(paths):
    """Build a reference voiceprint from one or more clips of the genuine speaker.
    More / longer clips (30 s+ total, clean) give a more stable voiceprint."""
    embs = torch.stack([embed(load_audio(p)) for p in paths])
    return torch.nn.functional.normalize(embs.mean(dim=0), dim=0)


def cosine(a, b):
    return float(torch.dot(a, b))


def decide(score):
    """Map a cosine score to 'match' / 'uncertain' / 'mismatch'."""
    if score >= MATCH_THRESHOLD + UNCERTAIN_MARGIN:
        return "match"
    if score <= MATCH_THRESHOLD - UNCERTAIN_MARGIN:
        return "mismatch"
    return "uncertain"


def verify_clip(path, ref):
    """Compare a whole clip, plus per-window scores for a timeline.

    Returns dict:
      score    - cosine similarity for the whole clip
      decision - 'match' | 'uncertain' | 'mismatch'
      match    - bool (decision == 'match')
      timeline - list of (start_second, score) per window
      speaker_change - True if windows drop below the mismatch level for 3
                       consecutive windows (a speaker swap mid-call)
    """
    wav = load_audio(path)
    score = cosine(embed(wav), ref)

    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    timeline = []
    if len(wav) <= win:
        timeline.append((0.0, score))
    else:
        for start in range(0, len(wav) - win + 1, hop):
            s = cosine(embed(wav[start:start + win]), ref)
            timeline.append((start / SR, s))

    low = MATCH_THRESHOLD - UNCERTAIN_MARGIN
    run, change = 0, False
    for _, s in timeline:
        run = run + 1 if s <= low else 0
        if run >= 3:
            change = True
            break

    decision = decide(score)
    return {
        "score": score,
        "decision": decision,
        "match": decision == "match",
        "timeline": timeline,
        "speaker_change": change,
    }


# --- Calibration note ------------------------------------------------------
# The 0.25 threshold is a starting point. For your report, collect ~50 same-
# speaker pairs and ~50 different-speaker pairs (e.g. from VoxCeleb1 or
# LibriSpeech), compute cosine scores, and pick the threshold at the EER
# point. Report that number alongside your AASIST EER.
