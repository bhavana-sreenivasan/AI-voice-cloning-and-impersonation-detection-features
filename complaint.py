"""Build a pre-filled complaint draft (text + PDF) from a logged call. It does not submit anything."""
import datetime as dt

PORTALS = [
    ("National Cyber Crime Reporting Portal", "https://cybercrime.gov.in"),
    ("Cyber crime helpline (financial fraud)", "Call 1930"),
    ("Sanchar Saathi - Chakshu (report suspected fraud calls/SMS)", "https://sancharsaathi.gov.in"),
]


def _ts(t):
    return dt.datetime.fromtimestamp(t).strftime("%d %b %Y, %H:%M:%S")


def build_text(rec, complainant_name="", complainant_contact="", incident=""):
    ident = "not checked"
    if rec.get("identity_name"):
        ident = (f"closest saved voice '{rec['identity_name']}', similarity "
                 f"{rec['identity_score']:.2f} ({rec['identity_decision']})")
    p = rec.get("p_spoof")
    lines = [
        "DRAFT COMPLAINT - SUSPECTED VOICE-CLONE / IMPERSONATION CALL",
        f"Prepared: {dt.datetime.now().strftime('%d %b %Y, %H:%M')}",
        "",
        f"Complainant: {complainant_name or '[your name]'}",
        f"Contact: {complainant_contact or '[your phone / email]'}",
        "",
        f"Suspected caller number: {rec['phone']} (caller ID as received; it can be spoofed)",
        f"Date and time of call (as logged): {_ts(rec['ts'])}",
        "",
        "What happened:",
        incident or "[Describe what the caller said or asked for, and any money or information lost.]",
        "",
        "Automated analysis of the call audio:",
        f"- Overall risk: {rec['risk']} ({rec.get('reason') or 'n/a'})",
        f"- Voice authenticity: {rec.get('label') or 'n/a'}"
        + (f", average spoof probability {p:.0%}" if p is not None else ""),
        f"- Speaker identity: {ident}",
        f"- Additional verification: {rec.get('verification') or 'not performed'}",
        f"- Audio fingerprint (SHA-1): {rec.get('audio_sha1') or 'n/a'}",
    ]
    if rec.get("notes"):
        lines += ["", f"Notes: {rec['notes']}"]
    lines += [
        "",
        "Where to report:",
        *[f"- {n}: {u}" for n, u in PORTALS],
        "",
        "Note: this analysis is automated decision support, not proof. Keep the original recording, "
        "call logs and any payment details, and attach them to the complaint. Verify the portal "
        "details above before filing.",
    ]
    return "\n".join(lines)


def build_pdf(text):
    """Return PDF bytes, or None if reportlab is not installed."""
    try:
        from io import BytesIO
        from xml.sax.saxutils import escape
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        return None
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50)
    st = getSampleStyleSheet()
    story = []
    for i, line in enumerate(text.split("\n")):
        if not line.strip():
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(escape(line), st["Heading2"] if i == 0 else st["BodyText"]))
    doc.build(story)
    return buf.getvalue()
