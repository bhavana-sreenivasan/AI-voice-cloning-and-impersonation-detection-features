"""Cached model loading and the combined spoof + speaker check, shared by all pages."""
import os
import tempfile

import streamlit as st

import voice_store
from pipeline import analyze
from spoof import load_model


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_ref(name, _key):
    """Speaker embedding built from all saved clips of `name` (rebuilt when clips change)."""
    import speaker
    return speaker.enroll(voice_store.clips(name))


@st.cache_data(show_spinner=False)
def to_wav_cached(data_hash, data, ext):
    import audio_io
    return audio_io.to_wav(data, ext)
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
def run_checks(data_hash, data, ext, targets, lib_keys):
    """Spoof check + identity check against each target speaker. Cached per file + library state."""
    model, device = get_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        path = tmp.name
    idents, speaker_missing = {}, False
    try:
        res = analyze(path, model, device)
        if targets:
            try:
                import speaker
                for name in targets:
                    idents[name] = speaker.verify_clip(path, get_ref(name, lib_keys[name]))
            except ImportError:
                speaker_missing = True
    finally:
        os.remove(path)
    return res, idents, speaker_missing
