import os
import datetime
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import wave
import time
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

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

    total_frames = 0
    while not recording_queue.empty():
        data = recording_queue.get()
        total_frames += len(data)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())

    wf.close()
    duration = total_frames / samplerate
    print(f"[Save] Final audio duration: {duration:.2f} seconds")

# Transcribe live audio using Whisper and save with timestamps
def transcribe_live(model, transcript_filename):
    audio_buffer = np.empty((0, channels), dtype=np.float32)
    interval =  5 # seconds

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
                text = result["text"].strip()
                elapsed_time = time.time() - start_time
                timestamp = str(datetime.timedelta(seconds=elapsed_time))

                if text:
                    formatted = f"[{timestamp}] Speaker: {text}"
                    print(f"[Live] {formatted}")
                    with open(transcript_filename, "a") as f:
                        f.write(formatted + "\n")

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

# Local summarizer using Hugging Face BART
def summarize_text(text):
    if not text.strip():
        return "No transcript available to summarize."

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"[Summary] Error: {e}")
        return "Error generating summary."

# Main function
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
    transcribe_live(model, transcript_filename)

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

        # Format transcript with line breaks per segment
        segments = result.get("segments", [])
        transcript = "\n".join(
            [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Speaker: {seg['text'].strip()}" for seg in segments]
        ) if segments else result["text"].strip()

        if transcript:
            print("[Whisper] Final transcript:")
            print(transcript)

            with open(transcript_filename, "a") as f:
                f.write("\n[Final Transcript]\n" + transcript)

            print("[Assistant] Generating summary using local BART model...")
            summary = summarize_text(transcript)

            with open(summary_filename, "w") as f:
                f.write(summary)

        else:
            print("[Assistant] No transcript to summarize.")
            with open(transcript_filename, "a") as f:
                f.write("\n[Final Transcript] None\n")
            with open(summary_filename, "w") as f:
                f.write("")

        print(f"[Assistant] Saved:\n- {transcript_filename}\n- {summary_filename}")

if __name__ == "__main__":
    main()