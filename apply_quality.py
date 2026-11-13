"""apply_quality.py - add the narrowband (phone-quality) check to pipeline.py, live.py and app.py.

Run from the project folder (the one holding pipeline.py, live.py, app.py, quality.py):
    python apply_quality.py            # edit the files (a *.bak_quality backup is written first)
    python apply_quality.py --check    # only report what is applied / missing

Safe to run more than once: every edit is checked on its own, so a half-applied patch gets
completed and finished edits are skipped.  Line endings (LF / CRLF) are kept.
If an edit says NOT FOUND, that part of your file differs from the original; send me those lines.
"""
import os
import shutil
import sys

# (label, old text, new text, marker that proves the edit is already there)
PIPELINE = [
    ("import quality",
     "from detectors import load_model, forward_logits, pad_to as _pad, SR, MAX_LEN\n",
     "from detectors import load_model, forward_logits, pad_to as _pad, SR, MAX_LEN\nimport quality\n",
     "\nimport quality\n"),
    ("analyze(check_quality=...)",
     "def analyze(path, model, device):\n",
     "def analyze(path, model, device, check_quality=True):\n",
     "def analyze(path, model, device, check_quality=True):"),
    ("analyze: mark narrowband audio unreliable",
     '''        risk = risk_level(p_spoof)
    return {
        "label": label,''',
     '''        risk = risk_level(p_spoof)

    # Phone-quality (narrowband) audio: the detectors score loudness/pauses, not realness, so the
    # spoof score must not raise or clear an alert here. Raw values are kept under *_raw.
    q = quality.check(wav) if check_quality else {"hf_db": None, "narrowband": False}
    unreliable = bool(q["narrowband"])
    raw = {"label_raw": label, "risk_raw": risk, "alert_time_raw": alert_time}
    if unreliable:
        label, risk, alert_time = "Unreliable (phone-quality audio)", "Uncertain", None
    return {
        **raw,
        "unreliable": unreliable,
        "hf_db": q["hf_db"],
        "label": label,''',
     'unreliable = bool(q["narrowband"])'),
    ("fuse_risk signature + reason text",
     "def fuse_risk(spoof_risk, identity=None, speaker_change=False):",
     '''NARROWBAND_REASON = ("Phone-quality (narrowband) audio: the synthetic-voice detector is not reliable on it, "
                     "so it was not used. Rely on the voice-identity check or the challenge step.")


def fuse_risk(spoof_risk, identity=None, speaker_change=False, spoof_unreliable=False):''',
     "def fuse_risk(spoof_risk, identity=None, speaker_change=False, spoof_unreliable=False):"),
    ("fuse_risk: ignore the spoof score when unreliable",
     '''    Returns (risk, reason).
    """
    if spoof_risk == "High":''',
     '''    spoof_unreliable : True when the audio is narrowband (analyze()["unreliable"]): the spoof
                       result is ignored; identity mismatch / speaker change still raise High.
    Returns (risk, reason).
    """
    if spoof_unreliable:
        spoof_risk = None
    if spoof_risk == "High":''',
     "    if spoof_unreliable:\n        spoof_risk = None"),
    ("fuse_risk: return Uncertain (IMPORTANT: without this it falls through to 'Low')",
     '''    if spoof_risk == "Uncertain" or identity == "uncertain":''',
     '''    if spoof_unreliable:
        return "Uncertain", NARROWBAND_REASON
    if spoof_risk == "Uncertain" or identity == "uncertain":''',
     'return "Uncertain", NARROWBAND_REASON'),
]

APP = [
    ("app: pass spoof_unreliable to fuse_risk",
     '''    ident["speaker_change"] if ident else False,
)''',
     '''    ident["speaker_change"] if ident else False,
    spoof_unreliable=res.get("unreliable", False),
)''',
     'spoof_unreliable=res.get("unreliable", False)'),
]

LIVE = [
    ("LiveAnalyzer(check_quality=...)",
     "low=0.30, high=0.70, consec=3, gate=None, smooth=1):\n",
     "low=0.30, high=0.70, consec=3, gate=None, smooth=1, check_quality=False):\n",
     ("smooth=1, check_quality=False):", "smooth=1, check_quality=False, quality_windows=15):")),
    ("self.check_quality attribute",
     "        self.recent = deque(maxlen=max(1, smooth))\n        self.history = []\n",
     "        self.recent = deque(maxlen=max(1, smooth))\n        self.history = []\n"
     "        self.check_quality = check_quality   # True: narrowband audio never raises an alert (see quality.py)\n",
     "self.check_quality = check_quality"),
    ("narrowband windows do not count towards an alert",
     "        self.streak = self.streak + 1 if p > self.high else 0",
     '''        unreliable = False
        if self.check_quality:
            import quality
            unreliable = bool(quality.check(window, self.sr)["narrowband"])
        self.streak = self.streak + 1 if (p > self.high and not unreliable) else 0''',
     ('unreliable = bool(quality.check(window, self.sr)["narrowband"])', "self._hf_hist")),
    ("event: risk + unreliable flag",
     '''"risk": "High" if self.alert_active else risk_level(p, self.low, self.high),
              "new_alert": new_alert,''',
     '''"risk": "High" if self.alert_active else ("Uncertain" if unreliable else risk_level(p, self.low, self.high)),
              "unreliable": unreliable, "new_alert": new_alert,''',
     '"unreliable": unreliable, "new_alert": new_alert,'),
    ("summary: unrel share",
     '''        mean = float(np.mean(ps))
        return {"duration"''',
     '''        mean = float(np.mean(ps))
        unrel = float(np.mean([e.get("unreliable", False) for e in self.history]))
        return {"duration"''',
     "unrel = float(np.mean("),
    ("summary: unreliable_frac field",
     '''                "speaker_change": changed,
                "mean_latency_ms"''',
     '''                "speaker_change": changed, "unreliable_frac": unrel,
                "mean_latency_ms"''',
     '"unreliable_frac": unrel'),
    ("summary: risk",
     '''"risk": "High" if (first_alert is not None or changed) else risk_level(mean, self.low, self.high)}''',
     '''"risk": ("High" if (first_alert is not None or changed)
                         else "Uncertain" if unrel > 0.5 else risk_level(mean, self.low, self.high))}''',
     'else "Uncertain" if unrel > 0.5 else risk_level(mean, self.low, self.high)'),
]

FILES = [("pipeline.py", PIPELINE), ("live.py", LIVE), ("app.py", APP)]


def main():
    check_only = "--check" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    problems = 0
    if not os.path.isfile(os.path.join(here, "quality.py")):
        print("!! quality.py is not in this folder: copy it here (NOT into tests/), pipeline.py imports it.")
        problems += 1
    for fname, edits in FILES:
        path = os.path.join(here, fname)
        if not os.path.isfile(path):
            print(f"{fname}: file not found"); problems += 1; continue
        raw = open(path, "rb").read().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        changed = False
        print(f"\n{fname}" + ("  (CRLF line endings kept)" if crlf else ""))
        for label, old, new, marker in edits:
            if any(m in text for m in (marker if isinstance(marker, tuple) else (marker,))):
                print(f"  already applied : {label}")
            elif old in text:
                if check_only:
                    print(f"  MISSING         : {label}")
                else:
                    text = text.replace(old, new, 1); changed = True
                    print(f"  applied         : {label}")
            else:
                print(f"  NOT FOUND       : {label}   <- this part of {fname} differs from the original")
                problems += 1
        if changed:
            shutil.copy(path, path + ".bak_quality")
            out = text.replace("\n", "\r\n") if crlf else text
            open(path, "wb").write(out.encode("utf-8"))
    leftovers = [f for f in os.listdir(here) if f.endswith((".rej", ".orig"))]
    if leftovers:
        print("\nLeftover patch files you can delete:", ", ".join(leftovers))
    print("\nDone." if not problems else f"\n{problems} problem(s): see above.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())