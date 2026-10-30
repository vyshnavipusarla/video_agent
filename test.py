from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize_meeting, generate_title, notes_to_text

chunks = process_input("YOUR_YOUTUBE_URL_OR_FILE_PATH")
transcript = transcribe_all(chunks, language="english")  # or "hinglish"
print(generate_title(transcript))
print(notes_to_text(summarize_meeting(transcript)))