"""apply_gate_median.py - make the narrowband gate decide per CALL (median of recent windows), not per window.

Why: eval_callhome.py showed spliced / odd windows can read as wideband (extra clicks add energy at 4.5-7.5 kHz),
so a per-window gate let a few windows through.  A call's channel does not change from second to second:
the decision now uses the MEDIAN bandwidth reading of the last `quality_windows` (default 15) windows.

Run from the project folder, AFTER apply_quality.py:
    python apply_gate_median.py            # edit live.py (a live.py.bak_gate backup is written first)
    python apply_gate_median.py --check    # only report
Idempotent; keeps LF/CRLF line endings.  Reports NOT FOUND if your live.py differs from what apply_quality.py produces.
"""
import os
import shutil
import sys

EDITS = [
    ("LiveAnalyzer(quality_windows=...)",
     "smooth=1, check_quality=False):\n",
     "smooth=1, check_quality=False, quality_windows=15):\n",
     "quality_windows=15):"),
    ("history of recent bandwidth readings",
     "        self.check_quality = check_quality   # True: narrowband audio never raises an alert (see quality.py)\n",
     "        self.check_quality = check_quality   # True: narrowband audio never raises an alert (see quality.py)\n"
     "        self._hf_hist = deque(maxlen=max(1, quality_windows))   # recent bandwidth readings: the call's channel is decided on their median\n",
     "self._hf_hist = deque("),
    ("decide on the median of recent windows",
     '''        unreliable = False
        if self.check_quality:
            import quality
            unreliable = bool(quality.check(window, self.sr)["narrowband"])
''',
     '''        unreliable = False
        if self.check_quality:
            import quality
            hf = quality.hf_ratio_db(window, self.sr)
            if hf is not None:
                self._hf_hist.append(hf)
            # median of the recent windows: one spliced / odd window cannot flip the call's channel
            unreliable = bool(self._hf_hist) and float(np.median(self._hf_hist)) < quality.NARROWBAND_DB
''',
     "float(np.median(self._hf_hist)) < quality.NARROWBAND_DB"),
]


def main():
    check_only = "--check" in sys.argv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.py")
    if not os.path.isfile(path):
        print("live.py not found next to this script"); return 1
    raw = open(path, "rb").read().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    changed, problems = False, 0
    print("live.py" + ("  (CRLF kept)" if crlf else ""))
    for label, old, new, marker in EDITS:
        if marker in text:
            print(f"  already applied : {label}")
        elif old in text:
            if check_only:
                print(f"  MISSING         : {label}")
            else:
                text = text.replace(old, new, 1); changed = True
                print(f"  applied         : {label}")
        else:
            print(f"  NOT FOUND       : {label}   <- run apply_quality.py first, or this part of live.py differs")
            problems += 1
    if changed:
        shutil.copy(path, path + ".bak_gate")
        open(path, "wb").write((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    print("\nDone." if not problems else f"\n{problems} problem(s): see above.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())