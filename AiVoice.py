import os
import datetime
import openai
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import wave
import time
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load environment variables
load_dotenv()  # Load variables from .env into the environment
client = OpenAI()

# Create folders if they don't exist
os.makedirs("MeetingNotes", exist_ok=True)

# Recording config
samplerate = 16000
channels = 1
recording_queue = queue.Queue()
is_recording = True
start_time = None

# Record audio function
def record_audio():
    def callback(indata, frames, time_info, status):
        if status:
            print(f"[Live] Warning: {status}")
        recording_queue.put(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        print("[Assistant] Listening... Press Ctrl+C to stop.")
        while is_recording:
            sd.sleep(100)

# Save the recorded audio to a file
def save_audio(filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(samplerate)

    while not recording_queue.empty():
        data = recording_queue.get()
        wf.writeframes((data * 32767).astype(np.int16).tobytes())

    wf.close()

# Transcribe live audio using Whisper
def transcribe_live(model):
    audio_buffer = np.empty((0, channels), dtype=np.float32)
    interval = 5  # seconds

    def run_transcription():
        nonlocal audio_buffer
        while is_recording:
            time.sleep(interval)
            if len(audio_buffer) == 0:
                continue
            temp_audio = audio_buffer.copy()
            audio_buffer = np.empty((0, channels), dtype=np.float32)

            try:
                audio_path = "temp.wav"
                wf = wave.open(audio_path, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                wf.writeframes((temp_audio * 32767).astype(np.int16).tobytes())
                wf.close()

                result = model.transcribe(audio_path)
                print(f"[Live] {result['text'].strip()}")

            except Exception as e:
                print(f"[Live] Error: {e}")

    def buffer_audio():
        nonlocal audio_buffer
        while is_recording:
            try:
                data = recording_queue.get(timeout=1)
                audio_buffer = np.concatenate((audio_buffer, data))
            except queue.Empty:
                pass

    threading.Thread(target=buffer_audio, daemon=True).start()
    threading.Thread(target=run_transcription, daemon=True).start()

# Function to generate a summary using GPT-3.5
def summarize_text(text):
    if not text.strip():  # Check for empty text
        return "No transcript available to summarize."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You summarize meetings."},
                {"role": "user", "content": f"Summarize this meeting:\n{text}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Summary] Error: {e}")
        return "Error generating summary."

# Main function to handle the whole process
def main():
    global is_recording, start_time
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"recording_{now}.wav"
    transcript_filename = f"MeetingNotes/transcript_{now}.txt"
    summary_filename = f"MeetingNotes/summary_{now}.txt"

    print("[Assistant] Starting live transcription using local Whisper (Press Ctrl+C to stop)...")
    model = whisper.load_model("medium", device="cpu")

    start_time = time.time()

    # Start recording and transcription
    threading.Thread(target=record_audio, daemon=True).start()
    transcribe_live(model)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Assistant] Ctrl+C detected. Stopping recording and transcribing final audio with Whisper...")
        is_recording = False
        time.sleep(2)  # Allow buffer to flush

        print(f"[Assistant] Saving full audio to {audio_filename}...")
        save_audio(audio_filename)

        print("[Assistant] Transcribing final audio using Whisper...")
        result = model.transcribe(audio_filename)
        transcript = result["text"].strip()

        if transcript:
            print("[Whisper] Final transcript:")
            print(transcript)

            # Save the transcript to a file
            with open(transcript_filename, "w") as f:
                f.write(transcript)

            print("[Assistant] Generating summary using GPT...")
            summary = summarize_text(transcript)

            # Save the summary to a file
            with open(summary_filename, "w") as f:
                f.write(summary)

        else:
            print("[Assistant] No transcript to summarize.")
            with open(transcript_filename, "w") as f:
                f.write("")

            with open(summary_filename, "w") as f:
                f.write("")

        print(f"[Assistant] Saved:\n- {transcript_filename}\n- {summary_filename}")

if __name__ == "__main__":
    main()