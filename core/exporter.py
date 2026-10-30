from datetime import datetime

from fpdf import FPDF

# fpdf2's built-in fonts only cover Latin-1, so map common "smart" characters
# to plain equivalents and replace anything else that can't be drawn.
_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u2022": "-",
    "\u00a0": " ", "\u2192": "->",
}


def _safe(text: str) -> str:
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def _sections(notes) -> list:
    """Turn the notes object into [(heading, [lines])] for both exporters."""
    decisions = [f"- {d}" for d in notes.key_decisions] or ["- None recorded"]
    actions = [
        f"- {a.task} (Owner: {a.owner}; Deadline: {a.deadline})"
        for a in notes.action_items
    ] or ["- None recorded"]
    questions = [f"- {q}" for q in notes.open_questions] or ["- None recorded"]
    return [
        ("Summary", [notes.summary.strip()]),
        ("Key Decisions", decisions),
        ("Action Items", actions),
        ("Open Questions", questions),
    ]


def build_txt(title: str, notes, transcript: str, include_transcript: bool = True) -> str:
    lines = [title, "=" * len(title), f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ""]
    for heading, body in _sections(notes):
        lines += [heading.upper(), *body, ""]
    if include_transcript:
        lines += ["TRANSCRIPT", transcript.strip(), ""]
    return "\n".join(lines)


def build_pdf(title: str, notes, transcript: str, include_transcript: bool = True) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def write(text: str, size: int = 11, style: str = "", gap: float = 6):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, gap, _safe(text), new_x="LMARGIN", new_y="NEXT")

    write(title, size=18, style="B", gap=9)
    write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", size=9)
    pdf.ln(4)

    for heading, body in _sections(notes):
        write(heading, size=13, style="B", gap=8)
        for line in body:
            write(line)
        pdf.ln(3)

    if include_transcript:
        pdf.add_page()
        write("Transcript", size=13, style="B", gap=8)
        write(transcript.strip(), size=10, gap=5)

    return bytes(pdf.output())