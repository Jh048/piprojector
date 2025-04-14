import os
import time
import dotenv
import speech_recognition as sr
import whisper
import openai
from datetime import datetime

# Load environment variables
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure output directory exists
os.makedirs("MeetingNotes", exist_ok=True)

# Initialize Whisper model
model = whisper.load_model("base")

def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Summarize this transcript into key points."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message["content"]

def record_and_transcribe():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    audio_frames = []

    print("[Assistant] Starting live transcription using local Whisper (Press Ctrl+C to stop)...")
    print("[Assistant] Listening... Press Ctrl+C to stop.")

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                result = model.transcribe("temp.wav")
                text = result.get("text", "").strip()
                if text:
                    print(f"[Live] {text}")
                audio_frames.append(audio.get_wav_data())
    except KeyboardInterrupt:
        print("\n[Assistant] Ctrl+C detected. Stopping recording and transcribing final audio with Whisper...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"MeetingNotes/recording_{timestamp}.wav"
    transcript_filename = f"MeetingNotes/transcript_{timestamp}.txt"
    summary_filename = f"MeetingNotes/summary_{timestamp}.txt"

    print(f"[Assistant] Saving full audio to {audio_filename}...")
    with open(audio_filename, "wb") as f:
        for frame in audio_frames:
            f.write(frame)

    print("[Assistant] Transcribing final audio using Whisper...")
    result = model.transcribe(audio_filename)
    transcript_text = result.get("text", "").strip()

    if transcript_text:
        with open(transcript_filename, "w") as f:
            f.write(transcript_text)
        print(f"[Assistant] Transcript saved to {transcript_filename}")

        print("[Assistant] Summarizing with OpenAI...")
        summary = summarize_text(transcript_text)
        with open(summary_filename, "w") as f:
            f.write(summary)
        print(f"[Assistant] Summary saved to {summary_filename}")
    else:
        print("[Assistant] No transcript to summarize.")

if __name__ == "__main__":
    record_and_transcribe()