
import streamlit as st
try:
    import os
    import tempfile
    import pandas as pd

    from pipeline import analyze, LOW_T, HIGH_T
    from spoof import load_model

    st.set_page_config(page_title="AI Voice Clone Detector", page_icon="🎙️", layout="centered")
    st.title("🎙️ AI Voice Clone & Impersonation Detector")
    st.caption("Prototype: pretrained AASIST anti-spoofing model (CPU)")

    @st.cache_resource
    def get_model():
        return load_model()

    model, device = get_model()
    print("1: model ready", flush=True)

    uploaded = st.file_uploader(
    "Upload an audio clip",
    type=["wav", "flac", "mp3", "m4a", "ogg", "opus", "aac", "amr", "webm", "mp4"],
)

    if uploaded is not None:
        print("A: file received", uploaded.name, uploaded.size, flush=True)
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
        print("B: analyze done", res["duration"], len(res["timeline"]), flush=True)

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
            st.error("🚨 Mixed audio: real voice, then synthetic voice from about "
            f"{res['alert_time']:.0f} s onward.")
        print("C: metrics and alerts drawn", flush=True)

        st.subheader("Risk over time")
        df = pd.DataFrame(res["timeline"], columns=["time (s)", "spoof probability"]).set_index("time (s)")
        print("D: dataframe built", flush=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        print("E: matplotlib imported", flush=True)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(df.index, df["spoof probability"], marker="o")
        ax.set_ylim(0, 1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("spoof probability")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        print("F: chart drawn", flush=True)

        st.caption(f"Each point is a ~4 s segment. Uncertain zone: {LOW_T:.0%}-{HIGH_T:.0%} spoof probability. "
                   f"Audio length: {res['duration']:.1f} s.")
    else:
        st.info("Upload a clip to begin.")
except Exception as e:
    st.exception(e)
