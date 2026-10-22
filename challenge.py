"""Challenge-response verification for uncertain results.

The caller must say a fresh random phrase. We (1) check the words with speech-to-text and
(2) re-run the spoof and speaker checks on that new recording. A pre-recorded clip or a
pre-generated clone cannot know the phrase in advance. A live voice-conversion tool still
can say it, which is why the spoof and identity checks are re-run on the response too.
"""
import difflib
import re
import secrets
import time

WORDS = ["maple", "river", "lantern", "orange", "falcon", "silver", "garden", "window", "thunder",
         "pencil", "marble", "harbor", "violet", "candle", "meadow", "copper", "island", "velvet"]
DIGITS = "zero one two three four five six seven eight nine".split()
TTL = 90          # seconds the challenge stays valid
PASS_MATCH = 0.80  # phrase similarity needed to count as repeated correctly
FAIL_MATCH = 0.60  # below this the phrase was clearly not repeated


def make_challenge():
    words = [secrets.choice(WORDS) for _ in range(3)]
    digits = "".join(str(secrets.randbelow(10)) for _ in range(4))
    text = f"{words[0]} {words[1]} {' '.join(digits)} {words[2]}"
    return {"text": text, "issued": time.time()}


def expired(ch, now=None):
    return (now or time.time()) - ch["issued"] > TTL


def _tokens(s):
    """Lowercase words; spelled-out digits become digits; digit runs are joined ('4 8 2 7' == '4827')."""
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    out = []
    for w in s.split():
        out.append(str(DIGITS.index(w)) if w in DIGITS else w)
    joined, run = [], ""
    for t in out:
        if t.isdigit():
            run += t
        else:
            if run:
                joined.append(run)
                run = ""
            joined.append(t)
    if run:
        joined.append(run)
    return joined


def match_score(expected, heard):
    a, b = _tokens(expected), _tokens(heard)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


_asr = None


def transcribe(path):
    """Speech-to-text with faster-whisper (preferred) or openai-whisper. Raises ImportError if neither."""
    global _asr
    if _asr is None:
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel("base.en", device="cpu", compute_type="int8")
            _asr = lambda p: " ".join(s.text for s in m.transcribe(p, language="en")[0])
        except ImportError:
            import whisper  # openai-whisper
            m = whisper.load_model("base.en")
            _asr = lambda p: m.transcribe(p, language="en")["text"]
    return _asr(path)


def decide(match, spoof_risk, identity=None, speaker_change=False):
    """-> (outcome, risk, reason). outcome: Verified | Failed | Inconclusive"""
    if spoof_risk == "High":
        return "Failed", "High", "The response recording was detected as synthetic speech."
    if identity == "mismatch" or speaker_change:
        return "Failed", "High", "The response voice does not match the claimed speaker."
    if match < FAIL_MATCH:
        return "Failed", "High", ("The caller did not say the random phrase "
                                  "(possible replayed or pre-generated audio).")
    if match < PASS_MATCH or spoof_risk == "Uncertain" or identity == "uncertain":
        return "Inconclusive", "Uncertain", "Still borderline: repeat with a new phrase or use a callback."
    return "Verified", "Low", "Phrase repeated correctly; voice looks genuine" + (
        " and matches the claimed speaker." if identity == "match" else ".")
