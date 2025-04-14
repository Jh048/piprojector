import speech_recognition as sr
import openai
import datetime
import os
import time
from dotenv import load_dotenv

load_dotenv()  # This loads your .env file

openai.api_key = os.getenv("OPENAI_API_KEY")
# 🔑 Set your OpenAI API key here

class AIAssistantSecretary:
    def __init__(self, duration=60):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.duration = duration  # in seconds
        self.transcript = ""
        self.summary = ""
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def record_and_transcribe(self):
        print("[Assistant] Starting live transcription...")

        def callback(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio)
                self.transcript += " " + text
                print(f"[Live] {text}")
            except sr.UnknownValueError:
                print("[Live] (Could not understand audio)")
            except sr.RequestError as e:
                print(f"[Live] (API Error: {e})")

        # Start background listening
        stop_listening = self.recognizer.listen_in_background(self.mic, callback)

        # Countdown timer
        for remaining in range(self.duration, 0, -1):
            print(f"[Timer] {remaining:02d} seconds remaining...", end='\r')
            time.sleep(1)

        stop_listening(wait_for_stop=False)
        print("\n[Assistant] Recording complete.")
        return self.transcript

    def summarize(self):
        if not self.transcript.strip():
            return "[Assistant] No transcript available to summarize."

        print("[Assistant] Generating summary using OpenAI GPT...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful meeting assistant."},
                    {"role": "user", "content": f"Summarize this meeting:\n{self.transcript}"}
                ],
                temperature=0.5
            )
            self.summary = response['choices'][0]['message']['content']
            return self.summary
        except Exception as e:
            return f"[Assistant] Error generating summary: {e}"

    def save_files(self, folder="MeetingNotes"):
        os.makedirs(folder, exist_ok=True)
        transcript_file = os.path.join(folder, f"transcript_{self.timestamp}.txt")
        summary_file = os.path.join(folder, f"summary_{self.timestamp}.txt")

        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(self.transcript)

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(self.summary)

        print(f"[Assistant] Files saved:\n- {transcript_file}\n- {summary_file}")

    def run(self):
        self.record_and_transcribe()
        print("\n--- FINAL TRANSCRIPT ---\n", self.transcript)

        self.summarize()
        print("\n--- SUMMARY ---\n", self.summary)

        self.save_files()

# ✅ Optional: run this file directly to test recording and transcription
if __name__ == "__main__":
    assistant = AIAssistantSecretary(duration=120)  # Record for 2 minutes
    assistant.run()