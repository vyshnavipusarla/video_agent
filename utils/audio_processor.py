import os

import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download a YouTube video's audio and return the path of the WAV file."""
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "restrictfilenames": True,  # avoid special characters in filenames
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # The original extension varies (webm, m4a, opus, mp4...), but the
        # post-processed file is always .wav.
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    return filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16 kHz suits Whisper
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split a WAV file into fixed-length chunks and return their paths."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    base = os.path.splitext(wav_path)[0]

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    """Accept a YouTube URL or local file path; return a list of WAV chunk paths."""
    source = source.strip().strip('"')  # Windows "Copy as path" adds quotes

    if source.startswith(("http://", "https://")):
        print("Detected URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks


if __name__ == "__main__":
    src = input("YouTube URL or file path: ")
    for path in process_input(src):
        print(path)