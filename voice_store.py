"""Tiny on-disk voice library: voice_store/<speaker_name>/<clip files>."""
import os
import re
import shutil
import time

ROOT = "voice_store"


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:40].strip("_")


def list_speakers():
    if not os.path.isdir(ROOT):
        return []
    return sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))


def clips(name):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d))


def library_key(name):
    """Changes whenever the speaker's clips change (used as a cache key)."""
    return tuple((p, os.path.getmtime(p)) for p in clips(name))


def add_clip(name, data, ext):
    name = _safe(name)
    if not name:
        raise ValueError("Please enter a name.")
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{int(time.time())}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


def delete_speaker(name):
    shutil.rmtree(os.path.join(ROOT, name), ignore_errors=True)


def delete_clip(path):
    os.remove(path)
    d = os.path.dirname(path)
    if not os.listdir(d):
        os.rmdir(d)
