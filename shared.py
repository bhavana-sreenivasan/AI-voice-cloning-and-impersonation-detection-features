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
@st.cache_data(show_spinner=False)
def run_checks(data_hash, data, ext, targets, lib_keys, id_data=None, id_ext=".wav"):
    """Spoof check on the full recording; identity check on id_data (isolated voice) if given."""
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
        res = analyze(path, model, device)
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
