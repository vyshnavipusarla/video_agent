from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.summarizer import get_llm  # same Groq model setup as the summarizer
from core.vector_store import (
    build_vector_store,
    get_retriever,
    list_meetings,
    load_vector_store,
)

TOP_K = 4  # excerpts retrieved per question

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert meeting assistant. Answer the user's question using "
            "ONLY the transcript excerpts below. If the answer is not in the "
            "excerpts, say: \"I could not find this information in the meeting "
            "transcript.\" Be concise and precise, and do not invent names, "
            "numbers or dates.",
        ),
        (
            "human",
            "Previous conversation:\n{history}\n\n"
            "Transcript excerpts:\n{context}\n\n"
            "Question: {question}",
        ),
    ]
)

_answer_chain = None


def _get_answer_chain():
    """prompt | Groq LLM | string parser, built once."""
    global _answer_chain
    if _answer_chain is None:
        _answer_chain = (answer_prompt | get_llm() | StrOutputParser()).with_retry(
            stop_after_attempt=4
        )
    return _answer_chain


def format_docs(docs) -> str:
    return "\n\n".join(f"[Excerpt {i + 1}]\n{d.page_content}" for i, d in enumerate(docs))


def format_history(history: Optional[List[dict]]) -> str:
    """history: [{"role": "user" | "assistant", "content": "..."}]; last 6 turns used."""
    if not history:
        return "None"
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])


def build_rag(transcript: str):
    """Index a new transcript. Returns (vector_store, meeting_id)."""
    return build_vector_store(transcript)


def load_rag(meeting_id: str) -> Chroma:
    """Reopen a saved meeting."""
    return load_vector_store(meeting_id)


def ask_question(
    vector_store: Chroma, question: str, history: Optional[List[dict]] = None
) -> dict:
    """
    Answer a question about the meeting.

    Returns {"answer": str, "sources": [{"chunk_index": int, "text": str}, ...]}
    """
    if not question.strip():
        raise ValueError("Question is empty.")

    docs = get_retriever(vector_store, k=TOP_K).invoke(question)
    answer = _get_answer_chain().invoke(
        {
            "context": format_docs(docs),
            "question": question,
            "history": format_history(history),
        }
    )
    return {
        "answer": answer.strip(),
        "sources": [
            {"chunk_index": d.metadata.get("chunk_index"), "text": d.page_content}
            for d in docs
        ],
    }


if __name__ == "__main__":
    saved = list_meetings()
    store = None
    if saved:
        print(f"Saved meetings: {', '.join(saved)}")
        choice = input("Enter a meeting id to reopen, or press Enter for a new one: ").strip()
        if choice:
            store = load_rag(choice)

    if store is None:
        path = input("Path to a transcript .txt file: ").strip().strip('"')
        with open(path, "r", encoding="utf-8") as f:
            store, mid = build_rag(f.read())
        print(f"Meeting saved with id: {mid}")

    chat_history = []
    print("Ask questions about the meeting (blank line to quit).")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            break
        result = ask_question(store, q, chat_history)
        print(f"\nAssistant: {result['answer']}")
        chat_history += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": result["answer"]},
        ]