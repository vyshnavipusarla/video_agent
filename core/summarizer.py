import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()  # reads GROQ_API_KEY from .env

# Groq's model catalog changes often - override in .env if this one is retired.
# (llama-3.3-70b-versatile is no longer available on the free tier.)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Free-tier Groq has tight per-minute token limits, so long transcripts are
# summarised piece by piece (map-reduce) instead of in one huge request.
SINGLE_PASS_CHARS = 12000
CHUNK_CHARS = 8000
CHUNK_OVERLAP = 400


# ── Output schema ─────────────────────────────────────────────────────────────
class ActionItem(BaseModel):
    task: str = Field(description="What needs to be done")
    owner: str = Field(default="Unassigned", description="Person responsible, if mentioned")
    deadline: str = Field(default="Not specified", description="Due date, if mentioned")


class MeetingNotes(BaseModel):
    summary: str = Field(description="Concise summary of the meeting in 1-3 short paragraphs")
    key_decisions: List[str] = Field(description="Decisions that were made")
    action_items: List[ActionItem] = Field(description="Follow-up tasks")
    open_questions: List[str] = Field(
        default_factory=list,
        description="Unresolved questions or topics needing follow-up",
    )


# ── Prompts ───────────────────────────────────────────────────────────────────
chunk_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a meeting note-taker. Condense the transcript excerpt into "
            "bullet-point notes. Keep every decision, action item (with owner and "
            "deadline if stated), name, number and date. Do not invent anything.",
        ),
        ("human", "Transcript excerpt:\n\n{text}"),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert meeting assistant. From the meeting material, produce "
            "a concise summary, the key decisions made, the action items, and any "
            "open questions (unresolved topics needing follow-up). Use "
            "only what is in the material. If there are no decisions, action "
            "items or open questions, return an empty list. Use 'Unassigned' / 'Not specified' when "
            "an owner or deadline is not stated.",
        ),
        ("human", "Meeting material:\n\n{text}"),
    ]
)


# ── Chains (LCEL) ─────────────────────────────────────────────────────────────
_llm = None


def get_llm() -> ChatGroq:
    """Create the Groq chat model once and reuse it."""
    global _llm
    if _llm is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set in environment / .env")
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)
    return _llm


def _build_chains():
    llm = get_llm()
    chunk_chain = (chunk_prompt | llm | StrOutputParser()).with_retry(stop_after_attempt=4)
    final_chain = (final_prompt | llm.with_structured_output(MeetingNotes)).with_retry(
        stop_after_attempt=4
    )
    return chunk_chain, final_chain


_chains = None


def _get_chains():
    global _chains
    if _chains is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set in environment / .env")
        _chains = _build_chains()
    return _chains


# ── Public API ────────────────────────────────────────────────────────────────
def summarize_meeting(transcript: str) -> MeetingNotes:
    """Return summary, key decisions and action items for a transcript."""
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    chunk_chain, final_chain = _get_chains()

    if len(transcript) <= SINGLE_PASS_CHARS:
        return final_chain.invoke({"text": transcript})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP
    )
    pieces = splitter.split_text(transcript)
    print(f"Long transcript: summarising {len(pieces)} parts first...")

    # max_concurrency=1 keeps us under the free-tier rate limits.
    partial_notes = chunk_chain.batch(
        [{"text": p} for p in pieces], config={"max_concurrency": 1}
    )
    combined = "\n\n".join(
        f"[Part {i + 1}]\n{notes}" for i, notes in enumerate(partial_notes)
    )
    return final_chain.invoke({"text": combined})


title_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Based on the meeting transcript, generate a short professional meeting "
            "title (max 8 words). Only return the title, nothing else.",
        ),
        ("human", "{text}"),
    ]
)


def generate_title(transcript: str) -> str:
    """Return a short meeting title based on the start of the transcript."""
    title_chain = title_prompt | get_llm() | StrOutputParser()
    title = title_chain.with_retry(stop_after_attempt=3).invoke(
        {"text": transcript.strip()[:2000]}
    )
    return title.strip().strip('"')


def notes_to_text(notes: MeetingNotes) -> str:
    """Format the notes as plain text (for display and TXT/PDF export)."""
    lines = ["SUMMARY", notes.summary.strip(), "", "KEY DECISIONS"]
    lines += [f"- {d}" for d in notes.key_decisions] or ["- None recorded"]
    lines += ["", "ACTION ITEMS"]
    if notes.action_items:
        for a in notes.action_items:
            lines.append(f"- {a.task} (Owner: {a.owner}; Deadline: {a.deadline})")
    else:
        lines.append("- None recorded")
    lines += ["", "OPEN QUESTIONS"]
    lines += [f"- {q}" for q in notes.open_questions] or ["- None recorded"]
    return "\n".join(lines)


if __name__ == "__main__":
    path = input("Path to a transcript .txt file: ").strip().strip('"')
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(notes_to_text(summarize_meeting(text)))