"""Streamlit panel: build, download and track a complaint for one logged call."""
import streamlit as st

import call_log
import complaint


def render(call_id, key):
    rec = call_log.get(call_id)
    if rec is None:
        st.info("No logged call selected.")
        return
    c1, c2 = st.columns(2)
    name = c1.text_input("Your name", key=f"{key}_name")
    contact = c2.text_input("Your phone / email", key=f"{key}_contact")
    incident = st.text_area("What happened (optional)", key=f"{key}_incident",
                            placeholder="What did the caller say or ask for? Any money or information lost?")
    text = complaint.build_text(rec, name, contact, incident)
    with st.expander("Preview complaint draft", expanded=False):
        st.code(text, language=None)

    d1, d2 = st.columns(2)
    d1.download_button("Download .txt", text, file_name=f"complaint_{call_id}.txt", key=f"{key}_txt")
    pdf = complaint.build_pdf(text)
    if pdf:
        d2.download_button("Download PDF", pdf, file_name=f"complaint_{call_id}.pdf",
                           mime="application/pdf", key=f"{key}_pdf")
    else:
        d2.caption("Install `reportlab` for PDF export.")

    st.markdown("**File it here** (this app prepares the draft; you submit it yourself):")
    for label, target in complaint.PORTALS:
        if target.startswith("http"):
            st.link_button(label, target, key=f"{key}_{label}")
        else:
            st.write(f"- {label}: **{target}**")

    if rec["reported"]:
        st.success(f"Marked as reported" + (f" (ref: {rec['report_ref']})" if rec["report_ref"] else "."))
    else:
        ref = st.text_input("Complaint reference number (after filing)", key=f"{key}_ref")
        if st.button("Mark as reported", key=f"{key}_mark"):
            call_log.mark_reported(call_id, ref)
            st.rerun()
