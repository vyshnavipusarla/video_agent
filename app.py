import os
import re

import streamlit as st

from core.exporter import build_pdf, build_txt
from core.rag_engine import ask_question, build_rag
from core.summarizer import generate_title, summarize_meeting
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️", layout="wide")

if "meeting" not in st.session_state:
    st.session_state.meeting = None
if "chat" not in st.session_state:
    st.session_state.chat = []


def run_pipeline(source: str, language: str):
    """Audio -> transcript -> notes -> chat index. Stores results in session state."""
    with st.status("Processing meeting...", expanded=True) as status:
        st.write("Preparing audio...")
        chunks = process_input(source)

        st.write("Transcribing (this is the slow part)...")
        transcript = transcribe_all(chunks, language=language)
        if not transcript.strip():
            raise ValueError("No speech was detected in the audio.")

        st.write("Summarizing and extracting action items...")
        notes = summarize_meeting(transcript)
        title = generate_title(transcript)

        st.write("Building the chat index...")
        store, meeting_id = build_rag(transcript)

        status.update(label="Done!", state="complete", expanded=False)

    st.session_state.meeting = {
        "id": meeting_id,
        "title": title,
        "transcript": transcript,
        "notes": notes,
        "store": store,
    }
    st.session_state.chat = []


# ── Sidebar: input ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("New meeting")
    mode = st.radio("Audio source", ["YouTube URL", "Upload file"])
    language = st.radio(
        "Language",
        ["english", "hinglish"],
        help="Hinglish is transcribed and translated to English by Sarvam AI.",
    )
    if language == "hinglish":
        st.caption("Note: Hinglish audio is uploaded to Sarvam's servers.")

    source = None
    if mode == "YouTube URL":
        url = st.text_input("YouTube URL")
        if url.strip():
            source = url.strip()
    else:
        uploaded = st.file_uploader(
            "Audio or video file",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"],
        )
        if uploaded is not None:
            path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded.name))
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            source = path

    if st.button("Process meeting", type="primary", disabled=source is None):
        try:
            run_pipeline(source, language)
        except Exception as err:
            st.error(f"Something went wrong: {err}")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🎙️ AI Meeting Assistant")

meeting = st.session_state.meeting
if meeting is None:
    st.info("Add a YouTube URL or upload a file in the sidebar, then click **Process meeting**.")
    st.stop()

notes = meeting["notes"]
st.subheader(meeting["title"])

tab_summary, tab_transcript, tab_chat, tab_export = st.tabs(
    ["Summary", "Transcript", "Chat", "Export"]
)

with tab_summary:
    st.markdown("### Summary")
    st.write(notes.summary)

    st.markdown("### Key decisions")
    if notes.key_decisions:
        for d in notes.key_decisions:
            st.markdown(f"- {d}")
    else:
        st.caption("None recorded.")

    st.markdown("### Action items")
    if notes.action_items:
        st.dataframe(
            [
                {"Task": a.task, "Owner": a.owner, "Deadline": a.deadline}
                for a in notes.action_items
            ],
            hide_index=True,
        )
    else:
        st.caption("None recorded.")

    st.markdown("### Open questions")
    if notes.open_questions:
        for q in notes.open_questions:
            st.markdown(f"- {q}")
    else:
        st.caption("None recorded.")

with tab_transcript:
    st.text_area("Full transcript", meeting["transcript"], height=500)

with tab_chat:
    st.caption("Ask anything about this meeting. Answers come only from the transcript.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Transcript excerpts used"):
                    for s in msg["sources"]:
                        st.markdown(f"> {s['text']}")

    question = st.chat_input("Ask about the meeting...")
    if question:
        history = list(st.session_state.chat)
        st.session_state.chat.append({"role": "user", "content": question})
        try:
            result = ask_question(meeting["store"], question, history)
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )
        except Exception as err:
            st.session_state.chat.append(
                {"role": "assistant", "content": f"Sorry, that failed: {err}"}
            )
        st.rerun()

with tab_export:
    include_transcript = st.checkbox("Include full transcript", value=True)
    file_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", meeting["title"]).strip("_") or "meeting"

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download TXT",
            data=build_txt(meeting["title"], notes, meeting["transcript"], include_transcript),
            file_name=f"{file_stem}.txt",
            mime="text/plain",
        )
    with col2:
        st.download_button(
            "Download PDF",
            data=build_pdf(meeting["title"], notes, meeting["transcript"], include_transcript),
            file_name=f"{file_stem}.pdf",
            mime="application/pdf",
        )