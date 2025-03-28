#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import pygame

# Sound setup
pygame.mixer.init(frequency=44100, channels=2)

def generate_tone(frequency, duration_ms, volume=0.5):
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    tone = np.sin(2 * np.pi * frequency * t)
    tone = np.int16(tone * volume * 32767)
    stereo_tone = np.column_stack((tone, tone))  # Convert to stereo
    sound = pygame.sndarray.make_sound(stereo_tone)
    return sound

alarm_emergency = generate_tone(1000, 1000)
alarm_medium = generate_tone(700, 1000)
alarm_mild = generate_tone(400, 1000)

# Voice setup
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

def run_alarm(word, urgency):
    with open("alarm_log.txt", "a", encoding="utf-8") as log_file:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - {urgency.upper()} - {word}\n")

    color_map = {
        "emergency": ((0, 0, 255), "EMERGENCY", alarm_emergency, float('inf')),
        "medium": ((0, 255, 255), "MEDIUM", alarm_medium, 5),
        "mild": ((0, 255, 0), "MILD", alarm_mild, 2)
    }
    color, label, sound, repeat_count = color_map[urgency]
    white = (255, 255, 255)
    screen = np.zeros((480, 640, 3), dtype=np.uint8)

    count = 0
    while count < repeat_count or urgency == "emergency":
        for flash_color in [color, white]:
            screen[:] = flash_color
            cv2.putText(screen, f"{label}: {word.upper()}!!!", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.imshow('Eye Gaze Typing', screen)
            pygame.mixer.Sound.play(sound)
            speak(word)
            if cv2.waitKey(500) & 0xFF == 27:
                return "exit"
        if urgency != "emergency":
            count += 1
    return "done"

def play_alarm(word):
    if word in {"Suffocation", "Hospital"}:
        result = run_alarm(word, "emergency")
        if result == "exit":
            exit()
    elif word in {"Pain", "Oxygen", "Nurse", "Suction", "Thirsty", "Hungry", "Bad", "Head", "Hand", "Leg"}:
        run_alarm(word, "medium")
    else:
        run_alarm(word, "mild")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
LEFT_IRIS = 468
RIGHT_IRIS = 473

face_outline_pts = np.array([[250, 100], [200, 150], [200, 250],
                             [250, 300], [350, 300], [400, 250],
                             [400, 150], [350, 100]], np.int32)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Eye Gaze Typing', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Eye Gaze Typing', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

calibration_data = {"LEFT": [], "RIGHT": []}

def calibrate_direction(direction, duration=5):
    speak(f"Please look {direction.lower()} for {duration} seconds.")
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        overlay = frame.copy()
        cv2.polylines(overlay, [face_outline_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
                iris_pts = [LEFT_IRIS, RIGHT_IRIS]
                for idx in iris_pts:
                    pt = face_landmarks.landmark[idx]
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                lr = get_avg_gaze_ratio(face_landmarks.landmark, w, h)
                calibration_data[direction].append(lr)

        cv2.putText(frame, f"Calibrating {direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Eye Gaze Typing', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

def get_avg_gaze_ratio(landmarks, w, h):
    def calc_ratio(eye, iris_idx):
        eye_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye], dtype=np.int32)
        iris_pt = np.array([landmarks[iris_idx].x * w, landmarks[iris_idx].y * h], dtype=np.int32)
        h_ratio = (iris_pt[0] - eye_pts[0][0]) / (eye_pts[1][0] - eye_pts[0][0] + 1e-5)
        return h_ratio
    l_ratio = calc_ratio(LEFT_EYE, LEFT_IRIS)
    r_ratio = calc_ratio(RIGHT_EYE, RIGHT_IRIS)
    return (l_ratio + r_ratio) / 2

for direction in ["LEFT", "RIGHT"]:
    calibrate_direction(direction)

left_avg = np.mean(calibration_data["LEFT"])
right_avg = np.mean(calibration_data["RIGHT"])
threshold = (left_avg + right_avg) / 2
speak("Calibration complete. Start typing.")

categories = {
    "Medical": ["Suction", "Pain", "Hospital", "Nurse", "Oxygen", "Suffocation"],
    "Mood": ["Thirsty", "Hungry", "Bad", "Happy"],
    "Movement": ["Head", "Hand", "Leg"],
    "TV": ["On", "Off"],
    "Alphabet": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPC", "READ", "DEL1", "DEL", "SAVE", "MENU"]
}
top_level = list(categories.keys())

typed_text = ""
current_group = top_level.copy()
current_category = None
gaze_start_time = None
confirmed_direction = None
selected_word = None
fade_start_time = None
fade_duration = 1.5
awaiting_confirmation = False
pending_selection = None
confirmation_start_time = None
confirmation_required_categories = [k for k in categories if k != "Alphabet"]
previous_category = None

def subdivide_group(group):
    mid = len(group) // 2
    return group[:mid], group[mid:]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    overlay = frame.copy()
    cv2.polylines(overlay, [face_outline_pts], isClosed=True, color=(0, 255, 0), thickness=2)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    direction = "Straight"
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1)
            )
            iris_pts = [LEFT_IRIS, RIGHT_IRIS]
            for idx in iris_pts:
                pt = face_landmarks.landmark[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            gaze_ratio = get_avg_gaze_ratio(face_landmarks.landmark, w, h)
            if gaze_ratio < threshold - 0.05:
                direction = "LEFT"
            elif gaze_ratio > threshold + 0.05:
                direction = "RIGHT"

    current_time = time.time()

    if awaiting_confirmation:
        if direction in ["LEFT", "RIGHT"]:
            if confirmed_direction != direction:
                confirmation_start_time = current_time
                confirmed_direction = direction
            elif current_time - confirmation_start_time >= 3:
                if direction == "LEFT":
                    typed_text = pending_selection
                    speak(pending_selection)
                    play_alarm(pending_selection)
                else:
                    speak("Cancelled")
                current_group = top_level.copy()
                current_category = None
                awaiting_confirmation = False
                pending_selection = None
                gaze_start_time = None
                confirmed_direction = None
                time.sleep(0.5)
        else:
            confirmation_start_time = None
            confirmed_direction = None
    else:
        if direction in ["LEFT", "RIGHT"]:
            if confirmed_direction != direction:
                gaze_start_time = current_time
                confirmed_direction = direction
            elif current_time - gaze_start_time >= 2:
                speak(direction.lower())
                gaze_start_time = None
                confirmed_direction = None
                left_half, right_half = subdivide_group(current_group)
                current_group = left_half if direction == "LEFT" else right_half

                if len(current_group) == 1:
                    selected_word = current_group[0]
                    fade_start_time = time.time()

                    if selected_word in categories:
                        if selected_word == "Alphabet" and current_category != "Alphabet":
                            typed_text = ""
                        current_group = categories[selected_word]
                        current_category = selected_word
                    elif selected_word == "MENU":
                        current_group = top_level.copy()
                        current_category = None
                    else:
                        if current_category == "Alphabet":
                            if selected_word == "SPC":
                                typed_text += " "
                                speak("space")
                            elif selected_word == "READ":
                                speak(typed_text if typed_text else "Nothing typed")
                            elif selected_word == "DEL1":
                                typed_text = typed_text[:-1]
                                speak("deleted last letter")
                            elif selected_word == "DEL":
                                typed_text = ""
                                speak("cleared all")
                            elif selected_word == "SAVE":
                                with open("typed_texts.txt", "a", encoding="utf-8") as f:
                                    f.write(typed_text.strip() + "\n")
                                speak("text saved")
                            else:
                                typed_text += selected_word
                                speak(selected_word)
                            current_group = categories["Alphabet"]
                        elif current_category in confirmation_required_categories:
                            pending_selection = selected_word
                            awaiting_confirmation = True
                            speak(f"Are you sure you want {selected_word}? Look left for yes, right for no.")
                        else:
                            typed_text = selected_word
                            speak(selected_word)
                            play_alarm(selected_word)
                            current_group = top_level.copy()
                            current_category = None
                    time.sleep(0.5)
        else:
            confirmed_direction = None
            gaze_start_time = None

    font_scale = 0.8
    if awaiting_confirmation:
        cv2.putText(frame, f"Are you sure: {pending_selection}?", (50, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Left: Yes", (50, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        cv2.putText(frame, f"Right: No", (50, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
    else:
        group_left, group_right = subdivide_group(current_group)
        left_str = " ".join(group_left)
        cv2.putText(frame, f"Left: {left_str}", (50, h // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)

        if current_category == "Alphabet":
            half = len(group_right) // 2
            right_line1 = " ".join(group_right[:half])
            right_line2 = " ".join(group_right[half:])
            cv2.putText(frame, f"Right: {right_line1}", (50, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            cv2.putText(frame, f"       {right_line2}", (50, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        else:
            right_str = " ".join(group_right)
            cv2.putText(frame, f"Right: {right_str}", (50, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    cv2.putText(frame, f"Text: {typed_text}", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Gaze: {direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if selected_word and fade_start_time:
        elapsed = time.time() - fade_start_time
        if elapsed < fade_duration:
            alpha = 1 - (elapsed / fade_duration)
            overlay = frame.copy()
            cv2.putText(overlay, selected_word, (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            selected_word = None
            fade_start_time = None

    cv2.imshow('Eye Gaze Typing', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

