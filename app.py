"""Streamlit demo: upload audio -> spoof check + optional speaker check -> fused risk.

Run:  streamlit run app.py
"""
import os
import tempfile
import traceback
import pandas as pd
import streamlit as st

from pipeline import analyze, fuse_risk, LOW_T, HIGH_T
from spoof import load_model

st.set_page_config(page_title="AI Voice Clone Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ AI Voice Clone & Impersonation Detector")
st.caption("Prototype: pretrained AASIST anti-spoofing model + ECAPA-TDNN speaker verification (CPU)")

AUDIO_TYPES = ["wav", "flac"]


@st.cache_resource
def get_model():
    return load_model()


def save_temp(f):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
        tmp.write(f.getbuffer())
        return tmp.name


model, device = get_model()

uploaded = st.file_uploader("Upload an audio clip (wav / flac)", type=AUDIO_TYPES)

with st.expander("Speaker identity check (optional)"):
    st.write("Upload a clean 20-30 s recording of the person the caller claims to be.")
    ref_file = st.file_uploader("Reference voice", type=AUDIO_TYPES, key="ref")

if uploaded is None:
    st.info("Upload a clip to begin.")
    st.stop()

st.audio(uploaded)
tmp_path = save_temp(uploaded)
ref_path = save_temp(ref_file) if ref_file is not None else None

ident = None
try:
    with st.spinner("Analyzing..."):
        res = analyze(tmp_path, model, device)
        if ref_path is not None:
            try:
                import speaker  # imported lazily so the app still runs without speechbrain
                ref_emb = speaker.enroll([ref_path])
                ident = speaker.verify_clip(tmp_path, ref_emb)
            except ImportError:
                st.warning("Speaker check skipped: run `pip install speechbrain` to enable it.")
except Exception as e:
    st.error(f"Could not analyze this file: {e}")
    st.code(traceback.format_exc())
    st.stop()
finally:
    os.remove(tmp_path)
    if ref_path:
        os.remove(ref_path)

final_risk, reason = fuse_risk(
    res["risk"],
    ident["decision"] if ident else None,
    ident["speaker_change"] if ident else False,
)
icon = {"Low": "🟢", "Uncertain": "🟡", "High": "🔴"}

# --- Headline result -------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Result", res["label"])
c2.metric("Confidence", f"{res['confidence']:.0%}")
c3.metric("Risk level", f"{icon[final_risk]} {final_risk}")

if final_risk == "High":
    st.error(f"High risk: {reason}")
elif final_risk == "Uncertain":
    st.warning(f"Uncertain: {reason}")
else:
    st.success(f"Low risk: {reason}")

if res["alert_time"] is not None:
    st.error(f"⚠️ Voice swap alert: synthetic voice detected from about {res['alert_time']:.0f} s onward.")

# --- Spoof timeline --------------------------------------------------------
st.subheader("Risk over time")
df = pd.DataFrame(res["timeline"], columns=["time (s)", "spoof probability"]).set_index("time (s)")
(st.bar_chart if len(df) < 3 else st.line_chart)(df)
st.caption(f"Each point is a ~4 s segment. Uncertain zone: {LOW_T:.0%}-{HIGH_T:.0%} spoof probability. "
           f"Audio length: {res['duration']:.1f} s.")

# --- Speaker identity ------------------------------------------------------
if ident is not None:
    st.subheader("Speaker identity")
    d1, d2 = st.columns(2)
    d1.metric("Similarity to reference", f"{ident['score']:.2f}")
    d2.metric("Identity decision", ident["decision"].capitalize())
    if ident["speaker_change"]:
        st.error("⚠️ The speaker appears to change partway through this clip.")
    sdf = pd.DataFrame(ident["timeline"], columns=["time (s)", "similarity"]).set_index("time (s)")
    (st.bar_chart if len(sdf) < 3 else st.line_chart)(sdf)
    st.caption("Higher similarity = closer to the reference voice. "
               "Thresholds are a starting point and should be calibrated on your own data.")

st.caption("Risk levels are decision support, not proof.")
