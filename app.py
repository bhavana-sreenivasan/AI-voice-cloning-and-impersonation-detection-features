"""Streamlit demo: upload audio -> Real/Fake + confidence + risk level.

Run:  streamlit run app.py
"""
import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)
import os
import tempfile
import pandas as pd
import streamlit as st

from pipeline import analyze, LOW_T, HIGH_T
from spoof import load_model

st.set_page_config(page_title="AI Voice Clone Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ AI Voice Clone & Impersonation Detector")
st.caption("Prototype: pretrained AASIST anti-spoofing model (CPU)")


@st.cache_resource
def get_model():
    return load_model()


model, device = get_model()

uploaded = st.file_uploader("Upload an audio clip (wav / flac)", type=["wav", "flac"])

if uploaded is not None:
    st.audio(uploaded)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    try:
        with st.spinner("Analyzing..."):
            res = analyze(tmp_path, model, device)
    except Exception as e:
        st.error(f"Could not analyze this file: {e}")
        st.stop()
    finally:
        os.remove(tmp_path)

    risk = res["risk"]
    icon = {"Low": "🟢", "Uncertain": "🟡", "High": "🔴"}[risk]

    c1, c2, c3 = st.columns(3)
    c1.metric("Result", res["label"])
    c2.metric("Confidence", f"{res['confidence']:.0%}")
    c3.metric("Risk level", f"{icon} {risk}")

    if risk == "High":
        st.error("High risk: this voice is likely AI-generated or cloned.")
    elif risk == "Uncertain":
        st.warning("Uncertain: an additional verification step is recommended.")
    else:
        st.success("Low risk: this voice looks genuine.")
    if res["alert_time"] is not None:
        st.error(f"⚠️ Voice swap alert: synthetic voice detected from about {res['alert_time']:.0f} s onward.")
    st.subheader("Risk over time")
    df = pd.DataFrame(res["timeline"], columns=["time (s)", "spoof probability"]).set_index("time (s)")
    if len(df) < 3:
        st.bar_chart(df)
    else:
        st.line_chart(df)
    st.caption(f"Each point is a ~4 s segment. Uncertain zone: {LOW_T:.0%}-{HIGH_T:.0%} spoof probability. "
               f"Audio length: {res['duration']:.1f} s.")
else:
    st.info("Upload a clip to begin. Tip: use samples from your eval subset in data\\LA\\ASVspoof2019_LA_eval\\flac.")
