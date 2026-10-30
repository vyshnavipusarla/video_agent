# 🎙️ AI Meeting Assistant

Turn a meeting recording into a transcript, a structured summary, and a chat you can ask questions to.

Give it a **YouTube URL** or an **uploaded audio/video file**. It transcribes the audio, translates Hinglish to English if needed, then uses an LLM to produce a summary, key decisions, action items and open questions. You can also chat with the transcript and export everything as **PDF** or **TXT**.

## Features

- **Audio input:** YouTube URLs (via `yt-dlp`) or uploaded files (mp3, wav, m4a, mp4, mkv, webm, ogg, flac)
- **Transcription:** local [OpenAI Whisper](https://github.com/openai/whisper) for English audio
- **Hinglish support:** audio is transcribed and translated to English with the Sarvam AI API
- **Meeting notes:** summary, key decisions, action items (task, owner, deadline) and open questions, plus an auto-generated title
- **Chat with the meeting:** RAG pipeline using ChromaDB and HuggingFace embeddings; answers are based only on the transcript and show the excerpts they used
- **Export:** download results as PDF or TXT, with or without the full transcript
- **Streamlit UI** with tabs for Summary, Transcript, Chat and Export

## How it works

```
YouTube URL / file
      │
      ▼
audio_processor.py ── download, convert to 16 kHz mono WAV, split into 10-min chunks
      │
      ▼
transcriber.py ────── English → local Whisper
      │               Hinglish → Sarvam AI (25 s pieces)
      ▼
   transcript
      ├──► summarizer.py ── LangChain LCEL + Groq → summary, decisions, action items, open questions
      │                     (long transcripts are summarised part by part, then combined)
      └──► vector_store.py ─ split + embed + save in ChromaDB (one folder per meeting)
                │
                ▼
           rag_engine.py ── retrieve top excerpts → Groq answers from them only
                │
                ▼
             app.py (Streamlit)  ──►  exporter.py (PDF / TXT)
```

## Project structure

```
video_agent/
├── app.py                  # Streamlit app
├── requirements.txt
├── .env                    # your API keys (do not commit)
├── core/
│   ├── __init__.py
│   ├── transcriber.py      # Whisper + Sarvam transcription
│   ├── summarizer.py       # Groq summary, decisions, action items, title
│   ├── vector_store.py     # ChromaDB + embeddings, one index per meeting
│   ├── rag_engine.py       # chat over the transcript
│   └── exporter.py         # PDF and TXT export
├── utils/
│   ├── __init__.py
│   └── audio_processor.py  # download, convert, chunk audio
├── downloads/              # created automatically (YouTube audio)
├── uploads/                # created automatically (uploaded files)
└── vector_db/              # created automatically (saved chat indexes)
```

## Requirements

- Python 3.10 or newer
- [FFmpeg](https://ffmpeg.org/) installed and on your PATH (needed by Whisper, yt-dlp and pydub)
- A free [Groq API key](https://console.groq.com/keys)
- A [Sarvam AI](https://www.sarvam.ai/) API key (only if you use Hinglish audio)

## Setup

**1. Install FFmpeg** (Windows):

```
winget install Gyan.FFmpeg
```

Close and reopen your terminal afterwards, then check that `ffmpeg -version` and `ffprobe -version` both work.

**2. Create a virtual environment and install dependencies:**

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**3. Create a `.env` file** in the project root:

```
GROQ_API_KEY=your_groq_key

# Only needed for Hinglish audio
SARVAM_API_KEY=your_sarvam_key

# Optional overrides (defaults shown)
GROQ_MODEL=openai/gpt-oss-120b
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
```

## Run

From the project root, with the virtual environment active:

```
streamlit run app.py
```

Then in the sidebar:

1. Choose **YouTube URL** or **Upload file**
2. Choose **english** or **hinglish**
3. Click **Process meeting**

For a first test, use a short recording (2 to 3 minutes). Transcription is the slow step.

### Running the pieces on their own

Run these as modules from the project root:

```
python -m core.transcriber      # audio → transcript
python -m core.summarizer       # transcript .txt → notes
python -m core.rag_engine       # chat in the terminal
python utils\audio_processor.py # download / convert / chunk only
```

## Tips

- **Speed:** on a CPU, Whisper `small` can take about as long as the audio. Set `WHISPER_MODEL=base` (or `tiny`) in `.env` for faster, less accurate results. A CUDA GPU makes it much faster; check with `python -c "import torch; print(torch.cuda.is_available())"`.
- **Groq rate limits:** the free tier has tight per-minute token limits. Requests are retried automatically. If you still hit limits, use `GROQ_MODEL=openai/gpt-oss-20b` or lower `CHUNK_CHARS` in `core/summarizer.py`.
- **Groq models change:** if you get a "model not found" error, pick a current model from [console.groq.com/docs/models](https://console.groq.com/docs/models) and set `GROQ_MODEL`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'core'` | Run from the project root, and add an empty `__init__.py` to `core` and `utils` |
| `No module named 'pydub'` or `'audioop'` | `pip install pydub audioop-lts` (`audioop-lts` is needed on Python 3.13+) |
| `ffprobe and ffmpeg not found` | Install FFmpeg and restart your terminal so PATH refreshes |
| Groq `404 model_not_found` | Set `GROQ_MODEL` in `.env` to a current model |
| `SARVAM_API_KEY is not set` | Add it to `.env` (only needed for Hinglish) |
| YouTube "format not available" | Update yt-dlp (`pip install -U yt-dlp`) and install Deno (`winget install DenoLand.Deno`) |

## Limitations

- **Hinglish audio is uploaded to Sarvam's servers.** English audio stays on your machine, but transcript text is always sent to Groq for summaries and chat.
- **Past meetings are not reloaded in the UI.** Chat indexes are saved in `vector_db/`, but the transcript and notes live only in the browser session, so export before closing the page.
- **PDF export** uses a built-in font that supports basic Latin characters only. Other symbols appear as `?`.
- You choose English or Hinglish yourself; the language is not detected automatically.

## Tech stack

Python · Streamlit · OpenAI Whisper · Sarvam AI · LangChain (LCEL) · Groq · ChromaDB · HuggingFace sentence-transformers · yt-dlp · pydub · fpdf2
