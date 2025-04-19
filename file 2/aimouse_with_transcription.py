import cv2
import mediapipe as mp
import autopy
import numpy as np
import speech_recognition as sr
import whisper
import time
import subprocess

# Hand tracking setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Whisper model
model = whisper.load_model("base")

# Transcription setup
recognizer = sr.Recognizer()
mic = sr.Microphone()
transcription = []

# Screen size
screen_width, screen_height = autopy.screen.size()

# Run transcription in background
def run_transcription():
    global transcription
    print("Starting transcription... Press Ctrl+C to stop.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    try:
        while True:
            with mic as source:
                audio = recognizer.listen(source, phrase_time_limit=5)

            try:
                audio_data = audio.get_wav_data()
                result = model.transcribe(audio_data, fp16=False)
                text = result['text'].strip()
                if text:
                    transcription.append(text)
                    print("You said:", text)
            except Exception as e:
                print("Transcription error:", e)
    except KeyboardInterrupt:
        print("\nTranscription stopped.")

        # Save transcript
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(transcription))
        print("Transcript saved to transcript.txt")

        # Run summarizer
        subprocess.run(["python3", "text_summarizer_script.py", "transcript.txt"])


# Run AI mouse
def run_ai_mouse():
    cap = cv2.VideoCapture(0)
    hand_detector = mp_hands.Hands(max_num_hands=1)
    smoothing_factor = 0.2
    prev_x, prev_y = 0, 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hand_detector.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])

                screen_x = np.interp(x, [0, image.shape[1]], [0, screen_width])
                screen_y = np.interp(y, [0, image.shape[0]], [0, screen_height])
                smooth_x = prev_x + (screen_x - prev_x) * smoothing_factor
                smooth_y = prev_y + (screen_y - prev_y) * smoothing_factor
                autopy.mouse.move(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("AI Mouse + Transcription", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


# Run both in parallel
import threading

if __name__ == "__main__":
    mouse_thread = threading.Thread(target=run_ai_mouse)
    transcription_thread = threading.Thread(target=run_transcription)

    mouse_thread.start()
    transcription_thread.start()

    mouse_thread.join()
    transcription_thread.join()