import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import threading
import os
import speech_recognition as sr
import whisper
import torch
import argparse
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from transformers import pipeline
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# --- Globals ---
transcription = [""]
exit_flag = False

# ---------------------- Hand Tracking + Mouse ---------------------- #
def run_hand_mouse():
    wCam, hCam = 640, 480
    frameR, wframe, smoothening = 100, 120, 7
    plocX, plocY = 0, 0

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

    detector = htm.handDetector(maxHands=2)
    wScr, hScr = autopy.screen.size()
    pTime = 0

    def zoom_image(img, zoom_factor=1.2):
        h, w = img.shape[:2]
        centerX, centerY = w // 2, h // 2
        radiusX, radiusY = int(w / (2 * zoom_factor)), int(h / (2 * zoom_factor))
        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY
        cropped = img[minY:maxY, minX:maxX]
        return cv2.resize(cropped, (w, h))

    while not exit_flag:
        success, img = cap.read()
        img = zoom_image(img)
        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img)

        if lmList:
            x1, y1 = lmList[4][1:]
            fingers = detector.fingersUp()

            if fingers[0] == 0 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                autopy.mouse.move(clocX, clocY)
                plocX, plocY = clocX, clocY

            length, _, lineInfo = detector.findDistance(4, 8, img)
            if length < 17:
                autopy.mouse.click()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

# ---------------------- Real-Time Transcription ---------------------- #
def run_transcription(update_ui_callback=None):
    global transcription

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium")
    parser.add_argument("--energy_threshold", default=1000, type=int)
    parser.add_argument("--record_timeout", default=2, type=float)
    parser.add_argument("--phrase_timeout", default=3, type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse', type=str)
    args = parser.parse_args([])

    phrase_time = None
    data_queue = Queue()
    phrase_bytes = bytes()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Select microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break
    else:
        source = sr.Microphone(sample_rate=16000)

    model_name = args.model
    if model_name != "large":
        model_name += ".en"
    audio_model = whisper.load_model(model_name)

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData):
        data_queue.put(audio.get_raw_data())

    recorder.listen_in_background(source, record_callback, phrase_time_limit=args.record_timeout)
    print("Transcription model loaded.")

    while not exit_flag:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    phrase_bytes = bytes()
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                phrase_bytes += audio_data

                audio_np = np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                full_text = "\n".join(transcription)
                if update_ui_callback:
                    update_ui_callback(full_text)
                else:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(full_text)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

# ---------------------- Text Summarization ---------------------- #
def summarize_text(text, max_length=130, min_length=30, model_name='t5-small'):
    summarizer = pipeline("summarization", model=model_name)
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    summaries = summarizer(chunks, max_length=max_length, min_length=min_length, do_sample=False)
    return " ".join([s['summary_text'] for s in summaries])

# ---------------------- UI ---------------------- #
def create_ui():
    root = tk.Tk()
    root.title("Live Transcription")
    root.geometry("700x400")
    transcript_box = ScrolledText(root, font=("Arial", 12))
    transcript_box.pack(fill=tk.BOTH, expand=True)

    def update_transcript_box(text):
        transcript_box.delete(1.0, tk.END)
        transcript_box.insert(tk.END, text)

    return root, update_transcript_box

# ---------------------- Main Entry Point ---------------------- #
if __name__ == "__main__":
    try:
        ui_root, update_ui = create_ui()

        t1 = threading.Thread(target=run_hand_mouse)
        t2 = threading.Thread(target=run_transcription, args=(update_ui,))
        t1.start()
        t2.start()

        ui_root.mainloop()

        exit_flag = True
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        exit_flag = True
    finally:
        full_text = " ".join(transcription).strip()
        if full_text:
            print("\n--- Full Transcript ---\n", full_text)
            with open("transcript.txt", "w", encoding="utf-8") as f:
                f.write(full_text)

            print("\n--- Summary ---\n")
            try:
                summary = summarize_text(full_text)
                print(summary)
                with open("meeting_summary.txt", "w", encoding="utf-8") as f:
                    f.write(summary)
            except Exception as e:
                print("Summarization failed:", e)