import speech_recognition as sr
import whisper
import datetime
import os
import signal
import wave
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class AIAssistantSecretary:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.transcript = ""
        self.summary = ""
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audio_filename = f"recording_{self.timestamp}.wav"
        self.listening = True
        self.audio_data_chunks = []

        # Whisper model (local)
        self.whisper_model = whisper.load_model("base")  # or "tiny", "small", etc.

    def signal_handler(self, sig, frame):
        print("\n[Assistant] Ctrl+C detected. Stopping recording and transcribing final audio with Whisper...")
        self.listening = False

    def live_transcribe_and_record(self):
        print("[Assistant] Starting live transcription using local Whisper (Press Ctrl+C to stop)...")

        def callback(recognizer, audio):
            try:
                raw_audio = audio.get_wav_data()
                temp_filename = "temp.wav"
                with open(temp_filename, "wb") as f:
                    f.write(raw_audio)

                result = self.whisper_model.transcribe(temp_filename, language="en")
                text = result["text"].strip()

                print(f"[Live] {text}")
                self.transcript += " " + text

                os.remove(temp_filename)
            except Exception as e:
                print(f"[Live] Error: {e}")

            self.audio_data_chunks.append(audio)

        stop_listening = self.recognizer.listen_in_background(self.mic, callback)
        signal.signal(signal.SIGINT, self.signal_handler)

        print("[Assistant] Listening... Press Ctrl+C to stop.\n")
        try:
            while self.listening:
                signal.pause()
        finally:
            stop_listening(wait_for_stop=False)

        # Save combined audio
        print(f"[Assistant] Saving full audio to {self.audio_filename}...")
        if self.audio_data_chunks:
            with wave.open(self.audio_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio_data_chunks[0].sample_width)
                wf.setframerate(self.audio_data_chunks[0].sample_rate)
                for chunk in self.audio_data_chunks:
                    wf.writeframes(chunk.get_raw_data())
        else:
            print("[Assistant] No audio data recorded.")

    def transcribe_with_whisper(self):
        print("[Assistant] Transcribing final audio using Whisper...")
        try:
            result = self.whisper_model.transcribe(self.audio_filename, language="en")
            self.transcript = result["text"].strip()
            print("[Whisper] Final transcript:\n", self.transcript)
        except Exception as e:
            print(f"[Whisper] Error: {e}")

    def summarize(self):
        if not self.transcript.strip():
            print("[Assistant] No transcript to summarize.")
            return

        print("[Assistant] Generating summary using GPT...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful meeting assistant."},
                    {"role": "user", "content": f"Summarize this meeting:\n{self.transcript}"}
                ]
            )
            self.summary = response['choices'][0]['message']['content']
            print("[Summary] Done.\n", self.summary)
        except Exception as e:
            print(f"[Summary] Error: {e}")

    def save_files(self, folder="MeetingNotes"):
        os.makedirs(folder, exist_ok=True)
        transcript_path = os.path.join(folder, f"transcript_{self.timestamp}.txt")
        summary_path = os.path.join(folder, f"summary_{self.timestamp}.txt")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(self.transcript)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.summary)

        print(f"[Assistant] Saved:\n- {transcript_path}\n- {summary_path}")

    def run(self):
        self.live_transcribe_and_record()
        self.transcribe_with_whisper()
        self.summarize()
        self.save_files()


if __name__ == "__main__":
    assistant = AIAssistantSecretary()
    assistant.run()