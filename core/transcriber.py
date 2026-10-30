import os
import time

import requests
import whisper
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()  # must run before the os.getenv() calls below

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# Each chunk is sliced into 25s pieces (5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25
SARVAM_MAX_RETRIES = 3

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():
    """Load the Whisper model once and reuse it."""
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe one English chunk locally with Whisper."""
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe", language="en")
    return result["text"]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript.

    Retries on timeouts, connection errors and 429/5xx responses.
    NOTE: this audio is uploaded to Sarvam's servers (Whisper stays local).
    """
    headers = {"api-subscription-key": SARVAM_API_KEY}
    retryable = {429, 500, 502, 503, 504}

    for attempt in range(1, SARVAM_MAX_RETRIES + 1):
        try:
            with open(piece_path, "rb") as f:
                files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
                data = {"model": SARVAM_MODEL, "with_diarization": "false"}
                response = requests.post(
                    SARVAM_STT_TRANSLATE_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120,
                )
        except (requests.ConnectionError, requests.Timeout) as err:
            if attempt == SARVAM_MAX_RETRIES:
                raise
            print(f"  Network error ({err.__class__.__name__}), retrying...")
            time.sleep(2 * attempt)
            continue

        if response.status_code in retryable and attempt < SARVAM_MAX_RETRIES:
            print(f"  Sarvam returned {response.status_code}, retrying...")
            time.sleep(2 * attempt)
            continue

        if not response.ok:
            print(f"\nSarvam returned {response.status_code}")
            print(f"Response body: {response.text}\n")
            response.raise_for_status()

        return response.json().get("transcript", "")

    return ""


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam's sync API only accepts <=30s audio. Split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    texts = []
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  Sarvam piece {i + 1}/{total_pieces} ...")
            text = _send_to_sarvam(piece_path).strip()
            if text:
                texts.append(text)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(texts)


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  -> Whisper (local model)
    - hinglish -> Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe all chunks in order and return one transcript."""
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language).strip()
        if text:
            parts.append(text)

    print("Transcription complete.")
    return " ".join(parts)


if __name__ == "__main__":
    from utils.audio_processor import process_input

    src = input("YouTube URL or file path: ")
    lang = input("Language (english/hinglish): ").strip() or "english"
    print(transcribe_all(process_input(src), language=lang))