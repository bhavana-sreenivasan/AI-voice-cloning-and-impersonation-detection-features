"""Streamlit demo: ONE upload -> spoof check + speaker identity vs the saved voice library -> fused risk.
Every analysis is logged (with the caller number) and can be turned into a complaint draft.
Uncertain results trigger a challenge-response verification step.

Run:  streamlit run app.py        (other pages: Live monitor, Call log)
"""
import hashlib
import html
import os
import tempfile

import pandas as pd
import streamlit as st

import call_log
import challenge
import report_panel
import voice_store
from pipeline import fuse_risk, LOW_T, HIGH_T
from shared import run_checks, split_call, to_wav_cached
st.set_page_config(page_title="AI Voice Clone Detector", layout="centered")
st.title(" AI Voice Clone & Impersonation Detector")
st.caption("Prototype: pretrained AASIST anti-spoofing + ECAPA-TDNN speaker verification (CPU)")

AUDIO_TYPES = ["wav", "flac", "mp3", "m4a", "aac", "ogg"]
AUTO = "Auto (compare with everyone)"
icon = {"Low": "🟢", "Uncertain": "🟡", "High": "🔴"}
ss = st.session_state


def best_of(idents):
    name = max(idents, key=lambda k: idents[k]["score"]) if idents else None
    return name, (idents[name] if name else None)


# --- Sidebar: voice library -----------------------------------------------
speakers = voice_store.list_speakers()
with st.sidebar:
    st.header("Saved voices")
    if not speakers:
        st.caption("Empty. Analyse a clip and save it to start the library.")
    for sp in speakers:
        n = len(voice_store.clips(sp))
        c1, c2 = st.columns([3, 1])
        c1.write(f"**{sp}** · {n} clip{'s' if n != 1 else ''}")
        if c2.button("🗑", key=f"del_{sp}", help=f"Delete all clips of {sp}"):
            voice_store.delete_speaker(sp)
            st.rerun()
    st.caption("Voice recordings are biometric data. Store them only with the speaker's consent.")

# --- Input -----------------------------------------------------------------
in1, in2 = st.columns([3, 2])
uploaded = in1.file_uploader("Audio clip", type=AUDIO_TYPES)
claimed = in2.selectbox(
    "Caller claims to be",
    [AUTO] + speakers,
    help="Pick a saved speaker to verify against one person, or Auto to compare with all saved voices.",
) if speakers else AUTO
if not speakers:
    in2.caption("No saved voices yet. Save a clip after analysis to enable identity checks.")
phone_raw = in2.text_input("Caller number (optional)", placeholder="+91 98765 43210")
phone = call_log.normalize_number(phone_raw)
if phone is None:
    in2.warning("That doesn't look like a phone number; it will be logged as 'unknown'.")
    phone = "unknown"

if uploaded is None:
    st.info("Upload a clip to begin.")
    st.stop()

data = uploaded.getvalue()
data_hash = hashlib.sha1(data).hexdigest()      # hash of the ORIGINAL upload
ext = os.path.splitext(uploaded.name)[1] or ".wav"
try:
    data, ext = to_wav_cached(data_hash, data, ext)     # <- new: everything below now sees WAV
except ValueError as e:
    st.error(str(e))
    st.stop()

targets = tuple(speakers) if claimed == AUTO else (claimed,)
lib_keys = {s: voice_store.library_key(s) for s in targets}

st.audio(data)
call_mode = st.radio("Recording type", ["Phone call (2+ speakers)", "Single speaker"], horizontal=True)
a_data, a_hash, a_ext = data, data_hash, ext      # what actually gets analysed
if call_mode.startswith("Phone"):
    try:
        with st.spinner("Separating speakers..."):
            voices = split_call(data_hash, data, ext)
    except Exception as e:
        st.error(f"Could not separate speakers: {e}")
        st.stop()
    names = list(voices)
    who = names[0]
    if len(names) > 1:
        who = st.radio("Which voice is the caller?", names, horizontal=True,
                       format_func=lambda k: f"{k} ({voices[k][1]:.0f} s of speech)")
        with st.expander("Listen to each separated voice"):
            for k in names:
                st.caption(k)
                st.audio(voices[k][0])
    a_data, a_hash, a_ext = voices[who][0], f"{data_hash}:{who}", ".wav"

try:
    with st.spinner("Analyzing..."):
        res, idents, speaker_missing = run_checks(a_hash, a_data, a_ext, targets, lib_keys)
except Exception as e:
    st.error(f"Could not analyze this file: {e}")
    st.stop()

if speaker_missing:
    st.warning("Speaker check skipped: run `pip install speechbrain` to enable it.")

best_name, ident = best_of(idents)
base_risk, base_reason = fuse_risk(
    res["risk"],
    ident["decision"] if ident else None,
    ident["speaker_change"] if ident else False,
)

# --- Log the call (one row per clip + number; refreshed if the result changes) ----
row_key = (data_hash, phone)
sig = (claimed, base_risk, base_reason)
if ss.get("row_key") != row_key:
    ss.call_id = call_log.log_call(
        phone, "upload", base_risk, res["label"], res["p_spoof"],
        best_name, ident["score"] if ident else None, ident["decision"] if ident else None,
        base_reason, data_hash)
    ss.row_key, ss.sig = row_key, sig
elif ss.get("sig") != sig:
    call_log.update_result(ss.call_id, base_risk, base_reason, res["label"], res["p_spoof"], best_name,
                           ident["score"] if ident else None, ident["decision"] if ident else None)
    ss.sig = sig
call_id = ss.call_id

# A finished challenge-response replaces the borderline result.
verif = ss.get("verif") if ss.get("verif", {}).get("call_id") == call_id else None
final_risk, reason = (verif["risk"], verif["reason"]) if verif else (base_risk, base_reason)

# --- 1. Verdict ------------------------------------------------------------
with st.container(border=True):
    st.markdown(f"### {icon[final_risk]} {final_risk} risk")
    st.write(reason)
    if verif:
        st.caption(f"After verification ({verif['outcome']}). Before: {base_risk}.")
    if res["alert_time"] is not None:
        st.write(f"⚠️ Synthetic voice detected from about **{res['alert_time']:.0f} s** onward.")
    if ident and ident["speaker_change"]:
        st.write("⚠️ The speaker appears to change partway through the clip.")
    st.caption(f"Logged as call #{call_id} · {phone}")

# --- 2. Key numbers --------------------------------------------------------
def card(label, value, sub=""):
    st.caption(label)
    st.markdown(
        f"<div style='font-size:1.6rem;font-weight:600;line-height:1.25;"
        f"word-break:break-word'>{html.escape(str(value))}</div>",
        unsafe_allow_html=True,
    )
    st.caption(sub or "\u00a0")
m1, m2, m3 = st.columns(3)
with m1.container(border=True):
    card("Authenticity", res["label"], f"{res['confidence']:.0%} confidence")
with m2.container(border=True):
    if ident:
        card("Closest saved voice", best_name,
             f"{ident['decision'].capitalize()} · {ident['score']:.2f}")
    else:
        card("Closest saved voice", "—", "No saved voices yet")
with m3.container(border=True):
    card("Suspicious audio", f"{res['frac_suspicious']:.0%}", f"of {res['duration']:.0f} s clip")

# --- 3. Additional verification (only for borderline results) --------------
if base_risk == "Uncertain":
    with st.container(border=True):
        st.markdown("#### Additional verification")
        if verif:
            v1, v2 = st.columns(2)
            v1.metric("Outcome", verif["outcome"])
            v2.metric("Phrase match", f"{verif['match']:.0%}" if verif["match"] is not None else "—")
            if verif["heard"]:
                st.caption(f"Heard: “{verif['heard']}”  ·  Expected: “{verif['expected']}”")
        else:
            st.write("The result is borderline. Ask the caller to say a fresh random phrase; "
                     "it is transcribed and the voice checks are re-run on that recording.")
        ch = ss.get("challenge")
        if st.button("New challenge phrase" if ch else "Generate challenge phrase"):
            ss.challenge, ss.pop_hash = challenge.make_challenge(), None
            ss.pop("verif", None)
            st.rerun()
        ch = ss.get("challenge")
        if ch:
            st.markdown(f"## “{ch['text']}”")
            st.caption(f"Valid for {challenge.TTL} s from when it was generated. Digits can be spoken one by one.")
            reply = (st.audio_input("Record the caller's response") if hasattr(st, "audio_input")
                     else st.file_uploader("Upload the caller's response", type=["wav"], key="reply_up"))
            if reply is not None:
                rdata = reply.getvalue()
                rhash = hashlib.sha1(rdata).hexdigest()
                if ss.get("pop_hash") != rhash:  # process each recording once
                    ss.pop_hash = rhash
                    if challenge.expired(ch):
                        st.error("This challenge expired. Generate a new phrase.")
                    else:
                        with st.spinner("Checking the response..."):
                            r_res, r_idents, _ = run_checks(rhash, rdata, ".wav", targets, lib_keys)
                            _, r_ident = best_of(r_idents)
                            heard, match = None, None
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                tmp.write(rdata)
                            try:
                                heard = challenge.transcribe(tmp.name)
                                match = challenge.match_score(ch["text"], heard)
                            except ImportError:
                                st.warning("Speech-to-text not installed: `pip install faster-whisper`. "
                                           "Only the voice checks were re-run.")
                            finally:
                                os.remove(tmp.name)
                        outcome, v_risk, v_reason = challenge.decide(
                            1.0 if match is None else match, r_res["risk"],
                            r_ident["decision"] if r_ident else None,
                            r_ident["speaker_change"] if r_ident else False)
                        if match is None and outcome == "Verified":  # phrase could not be checked
                            outcome, v_risk = "Inconclusive", "Uncertain"
                            v_reason = "Voice checks passed, but the phrase could not be checked (no speech-to-text)."
                        ss.verif = {"call_id": call_id, "outcome": outcome, "risk": v_risk, "reason": v_reason,
                                    "match": match, "heard": heard, "expected": ch["text"]}
                        call_log.update_verification(call_id, outcome, v_risk, v_reason)
                        st.rerun()

# --- 4. Details in tabs ----------------------------------------------------
tab_time, tab_auth, tab_id, tab_save, tab_report = st.tabs(
    ["Timeline", "Authenticity", "Identity", "Save voice", "Report"])

with tab_time:
    sp_df = pd.DataFrame(res["timeline"], columns=["time (s)", "Spoof probability"]).set_index("time (s)")
    if ident:
        id_df = pd.DataFrame(ident["timeline"], columns=["time (s)", "Similarity to voice"]).set_index("time (s)")
        both = pd.concat([sp_df, id_df], axis=1).sort_index().interpolate(limit_direction="both")
    else:
        both = sp_df
    (st.bar_chart if len(both) < 3 else st.line_chart)(both, height=300)
    st.caption("One line per check on a shared time axis. High spoof probability means synthetic speech; "
               "a high similarity means the voice sounds like the saved speaker. "
               "Each point is a ~4 s segment.")

with tab_auth:
    a1, a2 = st.columns(2)
    a1.metric("Spoof probability (average)", f"{res['p_spoof']:.0%}")
    a2.metric("Segments flagged", f"{res['frac_suspicious']:.0%}")
    st.caption(f"Uncertain zone: {LOW_T:.0%}-{HIGH_T:.0%} spoof probability. "
               "Thresholds are a starting point and should be calibrated on your own data.")

with tab_id:
    if not idents:
        st.info("No saved voices to compare with. Save a clip from the **Save voice** tab.")
    else:
        table = pd.DataFrame(
            [{"Saved speaker": n, "Similarity": round(v["score"], 3), "Decision": v["decision"].capitalize()}
             for n, v in sorted(idents.items(), key=lambda kv: -kv[1]["score"])]
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("A match only says the voice sounds like the saved speaker. "
                   "A good clone can match too, so always read it together with the authenticity result.")

with tab_save:
    if res["risk"] != "Low":
        st.caption("Only clips with a low spoof risk can be saved, so a fake voice can't pollute a profile.")
    else:
        with st.form("save_form", clear_on_submit=True):
            default = best_name if (ident and ident["decision"] == "match") else ""
            name = st.text_input("Speaker name", value=default, placeholder="e.g. alice")
            consent = st.checkbox("The speaker has consented to storing this recording.")
            if st.form_submit_button("Save this clip"):
                if not consent:
                    st.error("Consent is required.")
                else:
                    try:
                        voice_store.add_clip(name, a_data, a_ext)
                        st.toast(f"Saved to {name}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

with tab_report:
    if final_risk == "Low":
        st.caption("This call looks low risk. You can still prepare a complaint if you believe it was fraudulent.")
    report_panel.render(call_id, key=f"rep_{call_id}")

st.caption("Risk levels are decision support, not proof.")
