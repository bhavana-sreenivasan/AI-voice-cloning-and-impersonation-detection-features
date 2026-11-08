"""Cached model loading and the combined spoof + speaker check, shared by all pages."""
import os
import tempfile

import streamlit as st

import voice_store
from pipeline import analyze
import detectors


@st.cache_resource
def get_model():
    return detectors.load_model()   # set env SPOOF_MODEL=ssl to use SSL-AASIST


@st.cache_resource
def get_ref(name, _key):
    """Speaker embedding built from all saved clips of `name` (rebuilt when clips change)."""
    import speaker
    return speaker.enroll(voice_store.clips(name))


@st.cache_data(show_spinner=False)
def to_wav_cached(data_hash, data, ext):
    import audio_io
    return audio_io.to_wav(data, ext)


@st.cache_data(show_spinner=False)      # was uncached: every click re-separated the speakers
def split_call(data_hash, data, ext):
    """-> {'Speaker A': (wav_bytes, seconds), ...} longest speaker first."""
    import diarize
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        parts = diarize.split(path)
    finally:
        os.remove(path)
    return {f"Speaker {chr(65 + i)}": (diarize.to_wav_bytes(p), len(p) / diarize.SR)
            for i, p in enumerate(parts)}


@st.cache_data(show_spinner=False)
def similarity_to_user(data_hash, _voices, me_name, me_key):
    """Cosine similarity of each separated voice to the user's saved voice -> {voice_name: score}."""
    import io

    import soundfile as sf
    import torch

    import speaker
    ref = get_ref(me_name, me_key)
    out = {}
    for k, (wav_bytes, _) in _voices.items():
        y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        out[k] = speaker.cosine(speaker.embed(torch.tensor(y[: 30 * sr])), ref)   # <= 30 s is plenty
    return out


@st.cache_data(show_spinner=False)
def run_checks(data_hash, data, ext, targets, lib_keys, id_data=None, id_ext=".wav", spoof_on_isolated=False):
    """Identity check on id_data (the isolated caller voice) if given, else on the full recording.

    Spoof check: on the full recording by default; on the isolated caller voice when
    spoof_on_isolated=True (phone-call mode: the user's own side must not be scored)."""
    model, device = get_model()
    paths = []

    def _write(b, e):
        with tempfile.NamedTemporaryFile(delete=False, suffix=e) as tmp:
            tmp.write(b)
        paths.append(tmp.name)
        return tmp.name

    idents, speaker_missing = {}, False
    try:
        path = _write(data, ext)
        id_path = _write(id_data, id_ext) if id_data is not None else path
        res = analyze(id_path if spoof_on_isolated else path, model, device)
        if targets:
            try:
                import speaker
                for name in targets:
                    idents[name] = speaker.verify_clip(id_path, get_ref(name, lib_keys[name]))
            except ImportError:
                speaker_missing = True
    finally:
        for p in paths:
            os.remove(p)
    return res, idents, speaker_missing
