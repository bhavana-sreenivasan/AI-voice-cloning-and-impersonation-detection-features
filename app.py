"""Streamlit demo: upload or record audio -> Real/Fake + confidence + risk level.

Run:  streamlit run app.py
"""
import os
import tempfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Voice Clone Detector", page_icon="🎙️", layout="centered")

try:
    from pipeline import analyze, LOW_T, HIGH_T
    from spoof import load_model

    st.title("🎙️ AI Voice Clone & Impersonation Detector")
    st.caption("Prototype: pretrained AASIST anti-spoofing model (CPU)")

    @st.cache_resource
    def get_model():
        return load_model()

    model, device = get_model()

    tab_up, tab_rec = st.tabs(["📁 Upload", "🎤 Record"])
    with tab_up:
        uploaded = st.file_uploader(
            "Upload an audio clip",
            type=["wav", "flac", "mp3", "m4a", "ogg", "opus", "aac", "amr", "webm", "mp4"],
        )
    with tab_rec:
        recorded = st.audio_input("Record your voice (5-10 seconds)", sample_rate=16000)

    audio_file = recorded or uploaded  # a recording wins if both are present

    if audio_file is not None:
        st.audio(audio_file)
        suffix = os.path.splitext(getattr(audio_file, "name", "") or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_file.getbuffer())
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
            if res["alert_time"] < 1:
                st.error("🚨 The whole clip looks synthetic.")
            else:
                st.error("🚨 Mixed audio: real voice, then synthetic voice from about "
                         f"{res['alert_time']:.0f} s onward.")

        st.subheader("Risk over time")
        df = pd.DataFrame(res["timeline"], columns=["time (s)", "spoof probability"]).set_index("time (s)")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(df.index, df["spoof probability"], marker="o")
        ax.set_ylim(0, 1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("spoof probability")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(f"Each point is a ~4 s segment. Uncertain zone: {LOW_T:.0%}-{HIGH_T:.0%} spoof probability. "
                   f"Audio length analyzed: {res['duration']:.1f} s (max 60 s).")
        st.caption("⚠️ The model is trained on clean speech. Music, noise, phone-call or heavily "
                   "compressed audio can give unreliable results. Risk levels are decision "
                   "support, not proof.")
    else:
        st.info("Upload a clip or record your voice to begin.")
except Exception as e:
    st.exception(e)
