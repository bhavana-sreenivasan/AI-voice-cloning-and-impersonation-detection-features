"""Decide which separated voice is the caller (the party to analyse) using the user's saved voice.

Pure Python, no models: the app computes cosine similarities and passes them in, so the decision
logic can be tested on its own.
"""


def pick_caller(sims, match_t, margin, default):
    """sims: {voice_name: cosine similarity to the USER's saved voice}.

    Returns (caller, user_voice, reason):
      ok         - one voice matches the user; the caller is the other one
      single     - fewer than two voices, nothing to separate
      no_match   - no voice sounds like the user, so we can't tell which is which -> `default`
      both_match - both voices sound like the user (probably one person split in two) -> `default`
    Callers that get anything but 'ok' should fall back to manual choice.
    """
    if len(sims) < 2:
        return default, None, "single"
    user_voice = max(sims, key=sims.get)
    if sims[user_voice] <= match_t - margin:
        return default, None, "no_match"
    others = {k: v for k, v in sims.items() if k != user_voice}
    caller = min(others, key=others.get)
    if sims[caller] >= match_t + margin:
        return default, None, "both_match"
    return caller, user_voice, "ok"
