import os
import uuid
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_ROOT = "vector_db"  # one sub-folder per meeting lives in here
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once (downloads ~90 MB on first use)."""
    global _embeddings
    if _embeddings is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
        )
    return _embeddings


def _meeting_dir(meeting_id: str) -> str:
    return os.path.join(CHROMA_ROOT, meeting_id)


def build_vector_store(
    transcript: str, meeting_id: Optional[str] = None
) -> Tuple[Chroma, str]:
    """
    Split, embed and save a transcript in its OWN folder so meetings never mix.

    Returns (vector_store, meeting_id). Keep the id to reload the meeting later.
    """
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    meeting_id = meeting_id or uuid.uuid4().hex[:12]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    print(f"Indexing {len(docs)} transcript chunks (meeting {meeting_id})...")

    store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=_meeting_dir(meeting_id),
    )
    return store, meeting_id


def load_vector_store(meeting_id: str) -> Chroma:
    """Reopen a previously saved meeting without re-embedding."""
    path = _meeting_dir(meeting_id)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No saved meeting found for id '{meeting_id}'.")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=path,
    )


def list_meetings() -> List[str]:
    """IDs of all saved meetings, newest first."""
    if not os.path.isdir(CHROMA_ROOT):
        return []
    ids = [
        d for d in os.listdir(CHROMA_ROOT)
        if os.path.isdir(os.path.join(CHROMA_ROOT, d))
    ]
    return sorted(ids, key=lambda d: os.path.getmtime(_meeting_dir(d)), reverse=True)


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})