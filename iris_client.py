from picamera2 import Picamera2
import cv2
import requests
import time
import RPi.GPIO as GPIO
import subprocess
import speech_recognition as sr
import threading
import signal
import sys


SERVER = "http://192.168.137.1:8000"

running = True
speech_process = None
speech_lock = threading.Lock()
recent_objects = {}
stop_event = threading.Event()
safety_thread = None
cleanup_done = False
mic_device_index = None
WAKE_WORD = "iris"
suppress_safety_until = 0.0
safety_cooldown_until = 0.0


picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()


TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)


def cleanup():
    global speech_process, cleanup_done

    if cleanup_done:
        return

    cleanup_done = True

    with speech_lock:
        if speech_process and speech_process.poll() is None:
            speech_process.terminate()
            speech_process.wait(timeout=2)
            speech_process = None

    try:
        picam2.stop()
    except Exception:
        pass

    GPIO.cleanup()


def handle_shutdown(signum=None, frame=None):
    global running

    if not running:
        return

    running = False
    stop_event.set()
    print("Stopping IRIS...")

    try:
        stop_speaking()
    except Exception:
        pass


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def get_distance():
    if stop_event.is_set():
        return 999.0

    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start = time.time()
    stop = time.time()

    while GPIO.input(ECHO) == 0 and not stop_event.is_set():
        start = time.time()

    while GPIO.input(ECHO) == 1 and not stop_event.is_set():
        stop = time.time()

    elapsed = stop - start
    distance = elapsed * 17150
    return round(distance, 2)


def stop_speaking():
    global speech_process

    with speech_lock:
        if speech_process and speech_process.poll() is None:
            speech_process.terminate()
            speech_process.wait(timeout=2)
        speech_process = None


def speak(text):
    global speech_process

    print("IRIS:", text)

    stop_speaking()

    with speech_lock:
        speech_process = subprocess.Popen(
            ["espeak", "-ven+f3", "-s", "145", "-p", "55", text],
            stderr=subprocess.DEVNULL
        )


def reserve_speech_window(seconds=4.0):
    global suppress_safety_until
    suppress_safety_until = max(suppress_safety_until, time.time() + seconds)


def reset_recent_objects():
    recent_objects.clear()


def listen():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone(device_index=mic_device_index) as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)

        text = recognizer.recognize_google(audio).lower().strip()
        print("You said:", text)
        return text
    except Exception as exc:
        print("Listen error:", exc)
        return ""


def capture_detection():
    frame = picam2.capture_array()
    _, img = cv2.imencode(".jpg", frame)

    response = requests.post(
        SERVER + "/detect",
        files={"file": ("frame.jpg", img.tobytes(), "image/jpeg")},
        timeout=10
    )
    response.raise_for_status()
    data = response.json()
    return data.get("detections", [])


def select_microphone():
    global mic_device_index

    try:
        names = sr.Microphone.list_microphone_names()
    except Exception as exc:
        print("Microphone list error:", exc)
        mic_device_index = None
        return

    print("Available microphones:")

    for index, name in enumerate(names):
        print(f"[{index}] {name}")

    preferred_keywords = [
        "usb",
        "mic",
        "microphone",
        "webcam",
        "seeed",
        "respeaker"
    ]

    for index, name in enumerate(names):
        lowered = name.lower()
        if any(keyword in lowered for keyword in preferred_keywords):
            mic_device_index = index
            print(f"Using microphone [{index}] {name}")
            return

    mic_device_index = None
    print("Using default microphone device")


def navigation_guidance(objects, distance):
    current_time = time.time()

    for obj in objects:
        name = obj["object"]
        pos = obj["position"]
        track_id = obj.get("id", -1)
        speed = obj.get("speed", 0)
        danger = obj.get("danger", False)

        if track_id in recent_objects and current_time - recent_objects[track_id] < 4:
            continue

        recent_objects[track_id] = current_time

        if danger and speed > 20:
            return "Fast vehicle approaching"

        if distance < 60:
            return "Stop. Obstacle very close"

        if distance < 120:
            if pos == "center":
                return f"{name} ahead. Stop or move sideways"

            if pos == "left":
                return f"{name} on left. Move slightly right"

            if pos == "right":
                return f"{name} on right. Move slightly left"

        if distance < 200:
            return f"{name} {pos} about {int(distance)} centimeters"

    return None


def summarize_objects(objects):
    if not objects:
        return "I do not detect any known object right now."

    position_order = ("left", "center", "right")
    grouped = {position: {} for position in position_order}

    for obj in objects:
        position = obj.get("position", "center")
        name = obj.get("object", "object")
        grouped.setdefault(position, {})
        grouped[position][name] = grouped[position].get(name, 0) + 1

    parts = []

    for position in position_order:
        counts = grouped.get(position, {})
        if not counts:
            continue

        labels = []
        for name, count in sorted(counts.items()):
            if count == 1:
                labels.append(name)
            else:
                labels.append(f"{count} {name}s")

        parts.append(f"{position}: " + ", ".join(labels))

    return "I can describe your surroundings. " + "; ".join(parts) + "."


def build_guidance_message(objects, distance):
    for obj in objects:
        name = obj["object"]
        pos = obj["position"]
        speed = obj.get("speed", 0)
        danger = obj.get("danger", False)

        if danger and speed > 20:
            return "Fast vehicle approaching"

        if distance < 60:
            return f"Stop. {name} is very close"

        if distance < 120:
            if pos == "center":
                return f"{name} ahead. Stop or move sideways"

            if pos == "left":
                return f"{name} on left. Move slightly right"

            if pos == "right":
                return f"{name} on right. Move slightly left"

        if distance < 200:
            return f"{name} {pos} about {int(distance)} centimeters"

    return None


def build_detection_response(objects, distance):
    summary = summarize_objects(objects)
    guidance = build_guidance_message(objects, distance)
    distance_text = f"The nearest object is about {int(distance)} centimeters away."

    if guidance:
        return f"{guidance}. {distance_text}. {summary}"

    return f"{distance_text}. {summary}"


def detect_and_describe():
    objects = capture_detection()
    distance = get_distance()

    if distance < 40:
        reserve_speech_window()
        speak(
            "Emergency stop. Something is extremely close. "
            f"The nearest object is about {int(distance)} centimeters away. "
            + summarize_objects(objects)
        )
        return True

    reserve_speech_window()
    speak(build_detection_response(objects, distance))
    return True


def safety_monitor():
    global safety_cooldown_until

    while running and not stop_event.is_set():
        try:
            objects = capture_detection()
            distance = get_distance()

            if time.time() < suppress_safety_until:
                time.sleep(0.2)
                continue

            if time.time() < safety_cooldown_until:
                time.sleep(0.2)
                continue

            if distance < 40:
                speak("Emergency stop. Something extremely close")
                safety_cooldown_until = time.time() + 3
                time.sleep(2)
                continue

            guidance = navigation_guidance(objects, distance)
            if guidance:
                speak(guidance)
                safety_cooldown_until = time.time() + 3
            else:
                time.sleep(0.2)
        except Exception as exc:
            print("Detection error:", exc)
            time.sleep(0.5)


def extract_wake_command(text):
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    if cleaned == WAKE_WORD:
        return ""

    prefixes = (
        WAKE_WORD + " ",
        "hey " + WAKE_WORD + " ",
        "hi " + WAKE_WORD + " ",
        "ok " + WAKE_WORD + " ",
        "okay " + WAKE_WORD + " ",
    )

    for prefix in prefixes:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()

    return None


def is_detection_command(command):
    if not command:
        return True

    detection_phrases = (
        "detect",
        "what do you see",
        "what is in front",
        "what's in front",
        "what is around",
        "what's around",
        "surroundings",
        "scan",
        "look ahead",
        "objects",
    )

    return any(phrase in command for phrase in detection_phrases)


def handle_voice_command(text):
    global running

    normalized = text.strip().lower()

    if normalized in {"stop", "iris stop", "stop iris", "stop system", "shutdown iris", "exit iris"}:
        stop_speaking()
        speak("Stopping IRIS")
        time.sleep(1)
        running = False
        stop_event.set()
        return True

    if normalized in {"stop talking", "stop speaking", "be quiet", "silence", "quiet"}:
        stop_speaking()
        reserve_speech_window(1.5)
        return True

    return False


def conversation_loop():
    speak("IRIS ready. Say IRIS to start.")

    while running and not stop_event.is_set():
        text = listen()

        if not text:
            continue

        if handle_voice_command(text):
            continue

        wake_command = extract_wake_command(text)

        if wake_command is None:
            print("Ignored speech without wake word")
            continue

        if wake_command == "":
            speak("Listening")
            wake_command = listen()

            if not wake_command:
                reset_recent_objects()
                detect_and_describe()
                continue

            if handle_voice_command(wake_command):
                continue

        if is_detection_command(wake_command):
            try:
                reset_recent_objects()
                detect_and_describe()
            except requests.exceptions.RequestException as exc:
                print("Detection request failed:", exc)
                speak("Detection server error")
            except Exception as exc:
                print("Detection summary failed:", exc)
                speak("Detection error")
            continue

        try:
            response = requests.post(
                SERVER + "/chat",
                json={"prompt": wake_command},
                timeout=30
            )

            print("Chat status:", response.status_code)
            print("Chat response:", response.text)

            response.raise_for_status()
            reply = response.json()["response"]
            speak(reply)
        except requests.exceptions.RequestException as exc:
            print("Chat request failed:", exc)
            speak("Chat server error")
        except Exception as exc:
            print("Chat parse failed:", exc)
            speak("Chat reply error")


def main():
    global safety_thread

    select_microphone()

    safety_thread = threading.Thread(target=safety_monitor, daemon=True)
    safety_thread.start()
    conversation_loop()
    stop_event.set()

    if safety_thread.is_alive():
        safety_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        handle_shutdown()
    finally:
        cleanup()
