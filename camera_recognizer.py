import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import time
import os

# -------------------------------------------------------------
# Native Windows SAPI Speech Engine (Pure Natural Voice, No Chimes)
# -------------------------------------------------------------
class NativeSpeechEngine:
    def __init__(self):
        self.speaker = None
        self.enabled = True
        try:
            import win32com.client
            self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
            self.speaker.Volume = 100
            self.speaker.Rate = 0  # 0 is natural human speed (-10 to +10)
            print("Windows Native Speech Synthesizer initialized successfully (Volume 100%).")
        except Exception as e:
            print(f"Speech Engine Warning: {e}")

    def speak(self, text, interrupt=False):
        if not self.enabled or not text or self.speaker is None:
            return
        try:
            # Flags: 1 = SVSFlagsAsync (non-blocking), 3 = Async + Purge (interrupts instantly)
            flags = 3 if interrupt else 1
            self.speaker.Speak(str(text), flags)
        except Exception as e:
            try:
                import win32com.client
                self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
                self.speaker.Volume = 100
                self.speaker.Rate = 0
                self.speaker.Speak(str(text), 1)
            except Exception:
                pass

voice = NativeSpeechEngine()

# -------------------------------------------------------------
# Landmark & Geometric Feature Extraction
# -------------------------------------------------------------
def compute_hand_geometry(landmarks, wrist, span):
    curls = [
        np.linalg.norm(landmarks[4] - landmarks[2]) / span,   # Thumb
        np.linalg.norm(landmarks[8] - landmarks[5]) / span,   # Index
        np.linalg.norm(landmarks[12] - landmarks[9]) / span,  # Middle
        np.linalg.norm(landmarks[16] - landmarks[13]) / span, # Ring
        np.linalg.norm(landmarks[20] - landmarks[17]) / span  # Pinky
    ]
    spreads = [
        np.linalg.norm(landmarks[4] - landmarks[8]) / span,   # Thumb-Index
        np.linalg.norm(landmarks[8] - landmarks[12]) / span,  # Index-Middle
        np.linalg.norm(landmarks[12] - landmarks[16]) / span, # Middle-Ring
        np.linalg.norm(landmarks[16] - landmarks[20]) / span  # Ring-Pinky
    ]
    v1 = landmarks[5] - wrist
    v2 = landmarks[17] - wrist
    normal = np.cross(v1, v2)
    norm_val = np.linalg.norm(normal)
    if norm_val > 1e-4:
        normal = normal / norm_val
    else:
        normal = np.zeros(3)

    return np.concatenate([curls, spreads, normal])

def extract_features(results):
    if results.pose_landmarks:
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        sh_dist = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
        sh_dist = max(sh_dist, 0.1)
        ref_x = (l_sh.x + r_sh.x) / 2.0
        ref_y = (l_sh.y + r_sh.y) / 2.0
    else:
        ref_x, ref_y, sh_dist = 0.5, 0.5, 0.25

    lh_features = np.zeros(21 * 3)
    lh_present = 0.0
    lh_pos = np.zeros(3)
    lh_extra = np.zeros(12)
    if results.left_hand_landmarks:
        lh_present = 1.0
        lm = results.left_hand_landmarks.landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm])
        wrist = pts[0]
        lh_pos = (wrist - np.array([ref_x, ref_y, 0.0])) / sh_dist
        rel_pts = pts - wrist
        span = np.max(np.linalg.norm(rel_pts, axis=1))
        span = max(span, 0.02)
        lh_features = (rel_pts / span).flatten()
        lh_extra = compute_hand_geometry(pts, wrist, span)

    rh_features = np.zeros(21 * 3)
    rh_present = 0.0
    rh_pos = np.zeros(3)
    rh_extra = np.zeros(12)
    if results.right_hand_landmarks:
        rh_present = 1.0
        lm = results.right_hand_landmarks.landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm])
        wrist = pts[0]
        rh_pos = (wrist - np.array([ref_x, ref_y, 0.0])) / sh_dist
        rel_pts = pts - wrist
        span = np.max(np.linalg.norm(rel_pts, axis=1))
        span = max(span, 0.02)
        rh_features = (rel_pts / span).flatten()
        rh_extra = compute_hand_geometry(pts, wrist, span)

    if lh_present and rh_present:
        inter_dist = np.linalg.norm(lh_pos - rh_pos)
    else:
        inter_dist = 2.0

    base_135 = np.concatenate([
        lh_features,
        rh_features,
        [lh_present, rh_present],
        lh_pos,
        rh_pos,
        [inter_dist]
    ])

    feat_159 = np.concatenate([base_135, lh_extra, rh_extra])

    has_hands = (lh_present > 0 or rh_present > 0)
    return feat_159, has_hands

# -------------------------------------------------------------
# Main Application Loop
# -------------------------------------------------------------
def run_recognition():
    print("=" * 65)
    print(" Indian Sign Language (ISL) Multi-Mode Real-Time Recognizer ")
    print("=" * 65)

    # 1. Load 50-Word Model
    words_50_labels, words_50_model, words_50_hints = [], None, {}
    if os.path.exists("handshape_labels.json") and os.path.exists("handshape_model.h5"):
        with open("handshape_labels.json", "r", encoding="utf-8") as f:
            words_50_labels = json.load(f)
        words_50_model = tf.keras.models.load_model("handshape_model.h5")
        print(f"Loaded Mode 1: 50 Core Words Model ({len(words_50_labels)} classes)")
    if os.path.exists("sign_hints.json"):
        with open("sign_hints.json", "r", encoding="utf-8") as f:
            words_50_hints = json.load(f)

    # 2. Load 100-Word Model
    words_100_labels, words_100_model, words_100_hints = [], None, {}
    if os.path.exists("words_100_labels.json") and os.path.exists("words_100_model.h5"):
        with open("words_100_labels.json", "r", encoding="utf-8") as f:
            words_100_labels = json.load(f)
        words_100_model = tf.keras.models.load_model("words_100_model.h5")
        print(f"Loaded Mode 2: 100 Extended Words Model ({len(words_100_labels)} classes)")
    if os.path.exists("words_100_hints.json"):
        with open("words_100_hints.json", "r", encoding="utf-8") as f:
            words_100_hints = json.load(f)

    # 3. Load 70-Sentence Model
    sent_70_labels, sent_70_model, sent_70_hints = [], None, {}
    if os.path.exists("sentence_labels.json") and os.path.exists("sentence_model.h5"):
        with open("sentence_labels.json", "r", encoding="utf-8") as f:
            sent_70_labels = json.load(f)
        sent_70_model = tf.keras.models.load_model("sentence_model.h5")
        print(f"Loaded Mode 3: 70 Sentences Model ({len(sent_70_labels)} classes)")
    if os.path.exists("sentence_hints.json"):
        with open("sentence_hints.json", "r", encoding="utf-8") as f:
            sent_70_hints = json.load(f)

    # Build Unified Multi-Model Catalog
    unified_catalog = {}
    if words_50_model is not None:
        for l in words_50_labels:
            unified_catalog[l] = {"vdir": "videos", "hint": words_50_hints.get(l, ""), "source": "50W", "type": "Sign"}
    if words_100_model is not None:
        for l in words_100_labels:
            unified_catalog[l] = {"vdir": "words_100_videos", "hint": words_100_hints.get(l, ""), "source": "100W", "type": "Sign"}
    if sent_70_model is not None:
        for l in sent_70_labels:
            unified_catalog[l] = {"vdir": "sentence_videos", "hint": sent_70_hints.get(l, ""), "source": "Sent", "type": "Sentence"}

    unified_guide_signs = []
    for l, meta in unified_catalog.items():
        vpath = os.path.join(meta["vdir"], f"{l}.mp4")
        if os.path.exists(vpath):
            unified_guide_signs.append(l)
    if not unified_guide_signs:
        unified_guide_signs = list(unified_catalog.keys())

    available_modes = []
    if words_50_model and words_100_model and sent_70_model:
        available_modes.append("ALL_COMBINED")
    if words_50_model is not None: available_modes.append("50_WORDS")
    if words_100_model is not None: available_modes.append("100_WORDS")
    if sent_70_model is not None: available_modes.append("70_SENTENCES")

    if not available_modes:
        print("Error: No models found. Please train at least one model.")
        return

    mode_idx = 0
    current_mode = available_modes[mode_idx]

    def get_mode_resources(m):
        if m == "ALL_COMBINED":
            return None, list(unified_catalog.keys()), {}, "", "ALL COMBINED (220 SIGNS)"
        elif m == "50_WORDS":
            return words_50_model, words_50_labels, words_50_hints, "videos", "50 CORE WORDS"
        elif m == "100_WORDS":
            return words_100_model, words_100_labels, words_100_hints, "words_100_videos", "100 EXTENDED WORDS"
        else:
            return sent_70_model, sent_70_labels, sent_70_hints, "sentence_videos", "70 SENTENCES"

    active_model, active_labels, active_hints, active_vdir, mode_display_name = get_mode_resources(current_mode)
    if current_mode == "ALL_COMBINED":
        guide_signs = unified_guide_signs
    else:
        guide_signs = [l for l in active_labels if os.path.exists(os.path.join(active_vdir, f"{l}.mp4"))]
        if not guide_signs:
            guide_signs = active_labels
    current_guide_idx = 0
    show_guide = True

    def get_guide_cap(sign_name, fallback_vdir):
        if sign_name in unified_catalog:
            vdir = unified_catalog[sign_name]["vdir"]
        else:
            vdir = fallback_vdir
        path = os.path.join(vdir, f"{sign_name}.mp4")
        if os.path.exists(path):
            return cv2.VideoCapture(path)
        return None

    def get_hint(sign_name):
        if sign_name in unified_catalog and unified_catalog[sign_name]["hint"]:
            return unified_catalog[sign_name]["hint"]
        return active_hints.get(sign_name, "Mimic the demonstrator in the reference video.")

    guide_cap = get_guide_cap(guide_signs[current_guide_idx], active_vdir)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # Fast DirectShow camera startup on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sentence = []
    last_added = ""
    last_add_time = 0
    top_preds = []
    stable_sign = ""
    stable_count = 0
    smoothed_probs = None
    smoothed_50 = None
    smoothed_100 = None
    smoothed_70 = None
    best_sign = ""
    clean_display_name = ""
    auto_add_enabled = True
    notification_msg = "Sign silently to build sentence | Press 's' to speak aloud | Bksp/'b' to undo"
    notification_time = time.time()

    prev_features = None
    latched_sign = ""
    latched_clean = ""
    is_locked = False
    prev_time = time.time()
    fps = 30.0

    print("\n" + "=" * 70)
    print(" Indian Sign Language (ISL) Recognizer - Interactive Mode")
    print("=" * 70)
    print(" Controls:")
    print("   's'              : Speak assembled sentence aloud in English")
    print("   BACKSPACE or 'b' : Delete / Undo last added word (e.g. fix mistranslations)")
    print("   'r'              : Replace last added word with current detected sign")
    print("   SPACE            : Manually add currently detected sign to sentence")
    print("   'a'              : Toggle Auto-Add (ON: auto-adds after 0.5s hold | OFF: manual Space)")
    print("   'c'              : Clear entire sentence transcript")
    print("   'm' or TAB       : Switch Mode (50 Core Words <-> 100 Extended <-> 70 Sentences)")
    print("   't'              : Test voice output")
    print("   'v'              : Toggle voice audio ON / MUTED")
    print("   'g'              : Show / Hide corner reference video")
    print("   'n' / 'p'        : Next / Previous guide sign in corner video")
    print("   'j'              : Jump +10 signs forward in reference guide")
    print("   'k'              : Save snapshot image")
    print("   'q' or ESC       : Quit")
    print("=" * 70 + "\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        smooth_landmarks=True
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Draw visual landmarks on hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
                )
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )

            features, has_hands = extract_features(results)

            # Frame-to-frame velocity to differentiate steady deliberate signs from hand transitions
            if prev_features is not None and has_hands:
                hand_motion = float(np.mean(np.abs(features[:126] - prev_features[:126])))
            else:
                hand_motion = 0.0
            prev_features = features.copy() if has_hands else None

            is_transitioning = (hand_motion > 0.032)
            status_text = "Bring hands into view to sign"
            is_matched_guide = False
            dwell_prog = 0.0

            if has_hands:
                if current_mode == "ALL_COMBINED":
                    # Simultaneous Parallel Inference across all 3 models
                    in_135 = np.expand_dims(features[:135], axis=0)
                    in_159 = np.expand_dims(features[:159], axis=0)
                    raw_50 = words_50_model(in_135, training=False).numpy()[0]
                    raw_100 = words_100_model(in_159, training=False).numpy()[0]
                    raw_70 = sent_70_model(in_135, training=False).numpy()[0]

                    if smoothed_50 is None:
                        smoothed_50 = raw_50.copy()
                        smoothed_100 = raw_100.copy()
                        smoothed_70 = raw_70.copy()
                    else:
                        smoothed_50 = 0.30 * raw_50 + 0.70 * smoothed_50
                        smoothed_100 = 0.30 * raw_100 + 0.70 * smoothed_100
                        smoothed_70 = 0.30 * raw_70 + 0.70 * smoothed_70

                    all_candidates = []
                    # Top candidates from 50 Core Words
                    for i in np.argsort(smoothed_50)[-2:]:
                        all_candidates.append((words_50_labels[i], float(smoothed_50[i]), "50W", "Sign"))
                    # Top candidates from 100 Extended Words
                    for i in np.argsort(smoothed_100)[-2:]:
                        all_candidates.append((words_100_labels[i], float(smoothed_100[i]), "100W", "Sign"))
                    # Top candidates from 70 Sentences
                    for i in np.argsort(smoothed_70)[-2:]:
                        all_candidates.append((sent_70_labels[i], float(smoothed_70[i]), "Sent", "Sentence"))

                    all_candidates.sort(key=lambda x: x[1], reverse=True)
                    top_preds = all_candidates[:3]
                    best_sign, best_conf, best_src, best_type = top_preds[0]
                elif active_model is not None:
                    expected_dim = active_model.input_shape[1]
                    input_feat = np.expand_dims(features[:expected_dim], axis=0)
                    raw_probs = active_model(input_feat, training=False).numpy()[0]

                    if smoothed_probs is None or len(smoothed_probs) != len(raw_probs):
                        smoothed_probs = raw_probs.copy()
                    else:
                        smoothed_probs = 0.30 * raw_probs + 0.70 * smoothed_probs

                    top_indices = np.argsort(smoothed_probs)[-3:][::-1]
                    src_tag = "50W" if current_mode == "50_WORDS" else ("100W" if current_mode == "100_WORDS" else "Sent")
                    stype = "Sentence" if current_mode == "70_SENTENCES" else "Sign"
                    top_preds = [(active_labels[i], float(smoothed_probs[i]), src_tag, stype) for i in top_indices]
                    best_sign, best_conf, best_src, best_type = top_preds[0]
                else:
                    top_preds = []
                    best_sign, best_conf, best_src, best_type = "", 0.0, "", ""

                if top_preds:
                    curr_target = guide_signs[current_guide_idx]
                    if best_sign == curr_target and best_conf > 0.40:
                        is_matched_guide = True

                    clean_display_name = best_sign.replace("_", " ")

                    if best_conf >= 0.38 and not is_transitioning:
                        if best_sign == stable_sign:
                            stable_count += 1
                        else:
                            stable_sign = best_sign
                            stable_count = 1
                            is_locked = False

                        dwell_target = 13  # ~0.4s steady hold
                        dwell_prog = min(1.0, stable_count / dwell_target)

                        if stable_count >= 6:
                            latched_sign = best_sign
                            latched_clean = clean_display_name

                        if stable_count >= dwell_target:
                            is_locked = True
                            status_text = f"LOCKED: {clean_display_name}! (Lower hands to add)"
                        elif auto_add_enabled and stable_count >= 3:
                            status_text = f"Hold: {clean_display_name} ({int(dwell_prog*100)}%)"
                        else:
                            status_text = f"{best_type} [{best_src}]: {clean_display_name} ({best_conf*100:.0f}%)"
                    elif is_transitioning:
                        status_text = "Moving hands..."
                        stable_count = max(0, stable_count - 1)
                    else:
                        status_text = "Analyzing hand gesture..."
                        stable_count = 0
                        is_locked = False
            else:
                # Hands lowered / released: Natural trigger for two-handed signing
                current_time = time.time()
                if auto_add_enabled and latched_clean and (stable_count >= 6 or is_locked):
                    if latched_sign != last_added or (current_time - last_add_time) > 1.8:
                        sentence.append(latched_clean)
                        last_added = latched_sign
                        last_add_time = current_time
                        notification_msg = f"Added: '{latched_clean}' | [s] to Speak | [b] to Undo"
                        notification_time = current_time
                        print(f"[+] Added sign: '{latched_clean}' | Sentence: {' '.join(sentence)}")
                        if len(sentence) > 8:
                            sentence = sentence[-8:]

                top_preds = []
                smoothed_probs = None
                smoothed_50 = None
                smoothed_100 = None
                smoothed_70 = None
                status_text = "Bring hands into view to sign"
                stable_count = 0
                is_locked = False
                latched_sign = ""
                latched_clean = ""
                best_sign = ""
                clean_display_name = ""
                if (time.time() - last_add_time) > 1.0:
                    last_added = ""

            # -------------------------------------------------------------
            # UI Render: Top Header Bar & Sentence Display
            # -------------------------------------------------------------
            cv2.rectangle(frame, (0, 0), (w, 52), (20, 20, 20), -1)

            if current_mode == "ALL_COMBINED":
                mode_color = (255, 120, 0)
            elif current_mode == "50_WORDS":
                mode_color = (0, 180, 255)
            elif current_mode == "100_WORDS":
                mode_color = (0, 220, 100)
            else:
                mode_color = (200, 100, 255)

            # Mode Pill
            mode_text = f"{mode_display_name} ['m']"
            cv2.rectangle(frame, (10, 8), (250, 44), mode_color, -1)
            cv2.putText(frame, mode_text, (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)

            # Auto-Add Pill
            a_bg = (0, 160, 70) if auto_add_enabled else (70, 70, 70)
            a_txt = "Auto-Add: ON ('a')" if auto_add_enabled else "Auto: OFF ('a')"
            cv2.rectangle(frame, (258, 8), (390, 44), a_bg, -1)
            cv2.putText(frame, a_txt, (264, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            # Voice Status Pill
            v_color = (0, 220, 150) if voice.enabled else (120, 120, 120)
            v_text = "Voice: Ready ('s')" if voice.enabled else "Voice: Muted ('v')"
            cv2.rectangle(frame, (398, 8), (510, 44), (40, 40, 40), -1)
            cv2.rectangle(frame, (398, 8), (510, 44), v_color, 1)
            cv2.putText(frame, v_text, (404, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, v_color, 1, cv2.LINE_AA)

            # Real-Time FPS calculation
            now_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now_time - prev_time, 1e-4))
            prev_time = now_time

            # Sentence Output Box
            sent_box_x = 518
            cv2.rectangle(frame, (sent_box_x, 8), (w - 85, 44), (32, 28, 28), -1)
            cv2.rectangle(frame, (sent_box_x, 8), (w - 85, 44), (0, 200, 255), 1)

            if sentence:
                sentence_str = " ".join(sentence)
                cv2.putText(frame, f"Sentence: {sentence_str}", (sent_box_x + 12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 220), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Sentence: [Empty - sign to add words]", (sent_box_x + 12, 31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1, cv2.LINE_AA)

            # Live FPS Badge
            cv2.rectangle(frame, (w - 80, 8), (w - 10, 44), (25, 25, 25), -1)
            cv2.rectangle(frame, (w - 80, 8), (w - 10, 44), (0, 255, 180), 1)
            cv2.putText(frame, f"{int(fps)}fps", (w - 74, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)

            # -------------------------------------------------------------
            # UI Render: Interactive Shortcuts Helper Sub-Bar
            # -------------------------------------------------------------
            cv2.rectangle(frame, (0, 52), (w, 78), (12, 12, 12), -1)
            helper_str = "[s] Speak | [Bksp / b] Delete Word | [r] Replace | [Space] Add Sign | [c] Clear"
            cv2.putText(frame, helper_str, (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 235, 255), 1, cv2.LINE_AA)

            # Notification Banner
            if notification_msg and (time.time() - notification_time) < 2.8:
                n_w = 640
                nx1 = max(10, w//2 - n_w//2)
                nx2 = min(w - 10, w//2 + n_w//2)
                cv2.rectangle(frame, (nx1, 84), (nx2, 122), (0, 120, 220), -1)
                cv2.rectangle(frame, (nx1, 84), (nx2, 122), (255, 255, 255), 1)
                cv2.putText(frame, notification_msg, (nx1 + 12, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

            # -------------------------------------------------------------
            # UI Render: Bottom Status Bar
            # -------------------------------------------------------------
            bar_color = (0, 160, 0) if is_matched_guide else (20, 20, 20)
            cv2.rectangle(frame, (0, h - 85), (w, h), bar_color, -1)

            display_msg = f"MATCH: {status_text}!" if is_matched_guide else status_text
            cv2.putText(frame, display_msg, (15, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            # Hands-Free Dwell Progress Indicator
            if dwell_prog > 0:
                bar_x = 15
                bar_y = h - 22
                bar_w = 280
                bar_h = 10
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                bar_fill_color = (0, 255, 120) if is_locked else (0, 200, 255)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * dwell_prog), bar_y + bar_h), bar_fill_color, -1)

            # Top-3 Live Prediction Meters
            if top_preds:
                meter_x = w - 460
                for rank, pred_item in enumerate(top_preds):
                    cand = pred_item[0]
                    conf = pred_item[1]
                    src_tag = pred_item[2] if len(pred_item) > 2 else ""
                    y_offset = h - 65 + (rank * 22)
                    c_clean = cand.replace("_", " ")
                    if len(c_clean) > 13:
                        c_clean = c_clean[:11] + ".."
                    tag_str = f" [{src_tag}]" if src_tag else ""
                    cv2.putText(frame, f"{rank+1}. {c_clean}{tag_str}: {conf*100:.0f}%", (meter_x, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
                    bar_w = int(110 * conf)
                    cv2.rectangle(frame, (meter_x + 200, y_offset - 10), (meter_x + 310, y_offset), (50, 50, 50), -1)
                    color = (0, 255, 0) if cand == guide_signs[current_guide_idx] else (0, 200, 255)
                    cv2.rectangle(frame, (meter_x + 200, y_offset - 10), (meter_x + 200 + bar_w, y_offset), color, -1)

            # -------------------------------------------------------------
            # UI Render: Top-Right Reference Guide Box & Physical Hints
            # -------------------------------------------------------------
            if show_guide and guide_cap is not None:
                g_ret, g_frame = guide_cap.read()
                if not g_ret:
                    guide_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    g_ret, g_frame = guide_cap.read()

                if g_ret:
                    gw, gh = 280, 210
                    g_resized = cv2.resize(g_frame, (gw, gh))
                    gx = w - gw - 15
                    gy = 84
                    frame[gy:gy+gh, gx:gx+gw] = g_resized

                    box_border = (0, 255, 0) if is_matched_guide else (0, 200, 255)
                    cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), box_border, 3)

                    cv2.rectangle(frame, (gx, gy), (gx+gw, gy+28), (0, 0, 0), -1)
                    curr_guide_name = guide_signs[current_guide_idx]
                    clean_guide_name = curr_guide_name.replace("_", " ")
                    if len(clean_guide_name) > 16:
                        clean_guide_name = clean_guide_name[:14] + ".."

                    status_lbl = "MATCHED!" if is_matched_guide else f"{current_guide_idx+1}/{len(guide_signs)} {clean_guide_name}"
                    lbl_col = (0, 255, 0) if is_matched_guide else (0, 255, 255)
                    cv2.putText(frame, f"{status_lbl} (n/p/j)", (gx + 6, gy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, lbl_col, 1)

                    hint_text = get_hint(curr_guide_name)
                    cv2.rectangle(frame, (gx, gy+gh), (gx+gw, gy+gh+55), (25, 25, 25), -1)
                    cv2.rectangle(frame, (gx, gy+gh), (gx+gw, gy+gh+55), box_border, 1)

                    words_list = hint_text.split()
                    l1, l2, l3 = "", "", ""
                    for word in words_list:
                        if len(l1 + " " + word) < 32:
                            l1 = l1 + " " + word
                        elif len(l2 + " " + word) < 32:
                            l2 = l2 + " " + word
                        else:
                            l3 = l3 + " " + word
                    cv2.putText(frame, l1.strip(), (gx + 5, gy + gh + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    if l2:
                        cv2.putText(frame, l2.strip(), (gx + 5, gy + gh + 34),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    if l3:
                        cv2.putText(frame, l3.strip(), (gx + 5, gy + gh + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("Indian Sign Language (ISL) Recognizer - Multi-Mode", frame)

            # Key listener
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):
                # Speak entire sentence aloud
                if sentence:
                    full_text = " ".join(sentence)
                    notification_msg = f"Speaking: \"{full_text}\""
                    notification_time = time.time()
                    print(f"Speaking sentence aloud: {full_text}")
                    voice.speak(full_text, interrupt=True)
                else:
                    notification_msg = "Sentence is empty! Sign first, then press 's'."
                    notification_time = time.time()
                    voice.speak("Please sign to build your sentence first.", interrupt=True)
            elif key in (8, 127, ord('b'), ord('u')):  # Backspace, Delete, 'b', or 'u' (Undo)
                if sentence:
                    removed = sentence.pop()
                    last_added = sentence[-1].replace(" ", "_") if sentence else ""
                    last_add_time = time.time()
                    notification_msg = f"Undo: Deleted '{removed}'"
                    notification_time = time.time()
                    print(f"Undo: Deleted '{removed}'. Current sentence: {' '.join(sentence)}")
                else:
                    notification_msg = "Sentence is already empty."
                    notification_time = time.time()
            elif key == ord('r'):  # Replace last word with current sign
                if sentence and best_sign:
                    old_word = sentence.pop()
                    sentence.append(clean_display_name)
                    last_added = best_sign
                    last_add_time = time.time()
                    notification_msg = f"Replaced '{old_word}' -> '{clean_display_name}'"
                    notification_time = time.time()
                    print(f"Replaced '{old_word}' with '{clean_display_name}'")
                elif not sentence and best_sign:
                    sentence.append(clean_display_name)
                    last_added = best_sign
                    last_add_time = time.time()
                    notification_msg = f"Added: '{clean_display_name}'"
                    notification_time = time.time()
                else:
                    notification_msg = "No sign detected or nothing to replace."
                    notification_time = time.time()
            elif key == 32:  # SPACE - Manually add current sign
                if best_sign and has_hands:
                    sentence.append(clean_display_name)
                    last_added = best_sign
                    last_add_time = time.time()
                    notification_msg = f"Added: '{clean_display_name}'"
                    notification_time = time.time()
                    print(f"Manually added: '{clean_display_name}'")
                    if len(sentence) > 8:
                        sentence = sentence[-8:]
                else:
                    notification_msg = "Show a sign before pressing Space."
                    notification_time = time.time()
            elif key == ord('a'):  # Toggle Auto-Add
                auto_add_enabled = not auto_add_enabled
                state_lbl = "ON (Holds >= 0.5s auto-add)" if auto_add_enabled else "OFF (Manual - press Space)"
                notification_msg = f"Auto-Add: {state_lbl}"
                notification_time = time.time()
                print(f"Auto-Add: {state_lbl}")
            elif key == ord('c'):  # Clear
                sentence.clear()
                last_added = ""
                notification_msg = "Sentence cleared"
                notification_time = time.time()
                print("Sentence cleared.")
            elif key == ord('t'):  # Instant Voice Test
                notification_msg = "Voice test: Audio is working!"
                notification_time = time.time()
                print("Voice test triggered!")
                voice.speak("Voice test. Audio speech is working properly.", interrupt=True)
            elif key == ord('v'):
                voice.enabled = not voice.enabled
                notification_msg = f"Voice Speech: {'ON' if voice.enabled else 'MUTED'}"
                notification_time = time.time()
            elif key == ord('k'):
                snap_name = f"isl_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, frame)
                notification_msg = f"Saved: {snap_name}"
                notification_time = time.time()
                print(f"Snapshot saved to {snap_name}")
            elif key == ord('g'):
                show_guide = not show_guide
            elif key == ord('m') or key == 9:  # 'm' or TAB to cycle mode
                mode_idx = (mode_idx + 1) % len(available_modes)
                current_mode = available_modes[mode_idx]
                active_model, active_labels, active_hints, active_vdir, mode_display_name = get_mode_resources(current_mode)
                if current_mode == "ALL_COMBINED":
                    guide_signs = unified_guide_signs
                else:
                    guide_signs = [l for l in active_labels if os.path.exists(os.path.join(active_vdir, f"{l}.mp4"))]
                    if not guide_signs:
                        guide_signs = active_labels
                current_guide_idx = 0
                if guide_cap:
                    guide_cap.release()
                guide_cap = get_guide_cap(guide_signs[current_guide_idx], active_vdir)
                top_preds = []
                stable_sign = ""
                stable_count = 0
                smoothed_probs = None
                smoothed_50 = None
                smoothed_100 = None
                smoothed_70 = None
                notification_msg = f"Mode: {mode_display_name}"
                notification_time = time.time()
                print(f"Switched mode to: {mode_display_name} ({len(guide_signs)} guide signs)")
            elif key == ord('n'):
                current_guide_idx = (current_guide_idx + 1) % len(guide_signs)
                if guide_cap: guide_cap.release()
                guide_cap = get_guide_cap(guide_signs[current_guide_idx], active_vdir)
            elif key == ord('p'):
                current_guide_idx = (current_guide_idx - 1) % len(guide_signs)
                if guide_cap: guide_cap.release()
                guide_cap = get_guide_cap(guide_signs[current_guide_idx], active_vdir)
            elif key == ord('j'):
                current_guide_idx = (current_guide_idx + 10) % len(guide_signs)
                if guide_cap: guide_cap.release()
                guide_cap = get_guide_cap(guide_signs[current_guide_idx], active_vdir)

    if guide_cap:
        guide_cap.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
