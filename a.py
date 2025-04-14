#aimouse.py
import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

#########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
#########################

def run_virtual_mouse():
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(1)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()

    while True:
        # 1. Find hand Landmarks
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[4][1:]
            x2, y2 = lmList[8][1:]

            # 3. Check which fingers are up
            fingers = detector.fingersUp()

            # 4. Only Index Finger: Moving Mode
            if fingers[0] == 0 and fingers[2] == 0:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse
                autopy.mouse.move(clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 8. Both Index and middle fingers are up: Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # 9. Find distance between fingers
                length, img, lineInfo = detector.findDistance(4, 8, img)

                # 10. Click mouse if distance short
                if length < 15:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # 12. Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    run_virtual_mouse()

AiAssistant.py
import speech_recognition as sr
import openai
import datetime
import os
import time

# 🔑 Set your OpenAI API key here
openai.api_key = "***REMOVED***4d5DIuTelZQ-xnDrzwTBAQu8bHkgfT66YlkgQyjKk6NtQRXbwyiA2t9IxRRRcna3iPTHWCZzlgT3BlbkFJOG7hfncP8rbvqTcuuNMPnTI67CmwyWqIFbgpKu1D_ceBpZR1BVKl-OB_Hpbr3YIHgIlOkE7wMA"

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