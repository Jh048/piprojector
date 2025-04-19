import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import threading
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import argparse
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

# --- Global for cross-thread access ---
transcription = [""]
exit_flag = False

# ---------------------- Hand Tracking + Mouse ---------------------- #
def run_hand_mouse():
    wCam, hCam = 640, 480
    frameR = 100
    wframe = 120
    smoothening = 7

    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    pTime = 0
    detector = htm.handDetector(maxHands=2)
    wScr, hScr = autopy.screen.size()

    def zoom_image(img, zoom_factor=0.5):
        h, w = img.shape[:2]
        centerX, centerY = w // 2, h // 2
        radiusX, radiusY = int(w / (2 * zoom_factor)), int(h / (2 * zoom_factor))
        minX, maxX = centerX - radiusX, centerX + radiusX
        minY, maxY = centerY - radiusY, centerY + radiusY
        cropped = img[minY:maxY, minX:maxX]
        return cv2.resize(cropped, (w, h))

    while not exit_flag:
        success, img = cap.read()
        img = zoom_image(img, zoom_factor=1.2)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[4][1:]
            x2, y2 = lmList[8][1:]

            fingers = detector.fingersUp()
            cv2.rectangle(img, (wframe, frameR), (wCam - wframe, hCam - frameR), (255, 0, 255), 2)

            if fingers[0] == 0 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                autopy.mouse.move(clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            length, img, lineInfo = detector.findDistance(4, 8, img)
            if length < 17:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # cv2.imshow("Image", img)
        # cv2.waitKey(1)

# ---------------------- Real-Time Transcription ---------------------- #
def run_transcription():
    global transcription
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true')
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

    if 'linux' in platform:
        mic_name = args.default_microphone
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break
    else:
        source = sr.Microphone(sample_rate=16000)

    model_name = args.model
    if args.model != "large" and not args.non_english:
        model_name = model_name + ".en"
    audio_model = whisper.load_model(model_name)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    transcription = [""]

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Transcription model loaded.")

    while not exit_flag:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
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

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

# ---------------------- Text Summarization ---------------------- #
def summarize_text(text, max_length=130, min_length=30, model_name='t5-small'):
    from transformers import pipeline, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model_name)
    
    tokens = tokenizer.tokenize(text)
    max_tokens = tokenizer.model_max_length

    if len(tokens) > max_tokens:
        chunk_size = max_tokens - 50
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        for chunk in text_chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return ' '.join(summaries)
    else:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

# ---------------------- Main Entry Point ---------------------- #
if __name__ == "__main__":
    try:
        t1 = threading.Thread(target=run_hand_mouse)
        t2 = threading.Thread(target=run_transcription)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        exit_flag = True
        full_text = " ".join(transcription).strip()
        if full_text:
            print("\n--- Full Transcript ---\n")
            print(full_text)

            print("\n--- Summary ---\n")
            try:
                summary = summarize_text(full_text)
                print(summary)
                # Optionally save:
                with open("meeting_summary.txt", "w", encoding="utf-8") as f:
                    f.write(summary)
            except Exception as e:
                print("Summarization failed:", e)