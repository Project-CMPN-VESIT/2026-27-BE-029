import os
import json
import cv2
import numpy as np
import mediapipe as mp

TARGET_50 = [
    'Hello', 'Namaste', 'Good', 'Bad', 'No', 'Help', 'Come', 'Go', 'Eat', 'Drink',
    'Food', 'Milk', 'Book', 'Friend', 'Family', 'Father', 'Mother', 'Baby', 'Doctor', 'Nurse',
    'Hospital', 'Morning', 'Afternoon', 'Evening', 'Night', 'Day', 'Happy', 'Love', 'Like', 'Beautiful',
    'Hot', 'Cold', 'Big', 'Close', 'Give', 'Computer', 'Car', 'Bus', 'Laugh', 'Name',
    'Accept', 'Accident', 'Alone', 'Answer', 'Apple', 'Banana', 'Bread', 'Cat', 'Dog', 'Door'
]

def extract_hand_features(results):
    """
    Extracts instant, frame-by-frame hand shape + body-relative position.
    Features:
    - Left hand: 21 points relative to left wrist (normalized by palm size) = 63 values
    - Right hand: 21 points relative to right wrist (normalized by palm size) = 63 values
    - Hand presence flags: 2 values (left_present, right_present)
    - Left wrist pos relative to nose/shoulders: 3 values
    - Right wrist pos relative to nose/shoulders: 3 values
    - Inter-hand distance: 1 value
    Total: 63 + 63 + 2 + 3 + 3 + 1 = 135 features
    """
    # 1. Pose anchor points
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        sh_dist = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
        sh_dist = max(sh_dist, 0.1)
        ref_x = (l_sh.x + r_sh.x) / 2.0
        ref_y = (l_sh.y + r_sh.y) / 2.0
    else:
        ref_x, ref_y, sh_dist = 0.5, 0.5, 0.25

    # 2. Left Hand
    lh_features = np.zeros(21 * 3)
    lh_present = 0.0
    lh_pos = np.zeros(3)
    if results.left_hand_landmarks:
        lh_present = 1.0
        lm = results.left_hand_landmarks.landmark
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
        # Position relative to body center
        lh_pos = (wrist - np.array([ref_x, ref_y, 0.0])) / sh_dist
        
        # Hand shape relative to wrist
        rel_points = []
        for p in lm:
            rel_points.append([p.x - wrist[0], p.y - wrist[1], p.z - wrist[2]])
        rel_points = np.array(rel_points)
        # Scale normalization by palm/finger span
        span = np.max(np.linalg.norm(rel_points, axis=1))
        span = max(span, 0.02)
        lh_features = (rel_points / span).flatten()

    # 3. Right Hand
    rh_features = np.zeros(21 * 3)
    rh_present = 0.0
    rh_pos = np.zeros(3)
    if results.right_hand_landmarks:
        rh_present = 1.0
        lm = results.right_hand_landmarks.landmark
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
        # Position relative to body center
        rh_pos = (wrist - np.array([ref_x, ref_y, 0.0])) / sh_dist

        # Hand shape relative to wrist
        rel_points = []
        for p in lm:
            rel_points.append([p.x - wrist[0], p.y - wrist[1], p.z - wrist[2]])
        rel_points = np.array(rel_points)
        span = np.max(np.linalg.norm(rel_points, axis=1))
        span = max(span, 0.02)
        rh_features = (rel_points / span).flatten()

    # 4. Inter-hand distance
    if lh_present and rh_present:
        inter_dist = np.linalg.norm(lh_pos - rh_pos)
    else:
        inter_dist = 2.0

    feature_vec = np.concatenate([
        lh_features,
        rh_features,
        [lh_present, rh_present],
        lh_pos,
        rh_pos,
        [inter_dist]
    ])
    return feature_vec, (lh_present > 0 or rh_present > 0)

def main():
    print("Extracting per-frame hand shape and body-relative features...")
    mp_holistic = mp.solutions.holistic

    X = []
    y = []
    labels = []

    with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
        for idx, word in enumerate(TARGET_50):
            video_path = os.path.join("videos", f"{word}.mp4")
            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            step = 2 if fps > 40 else 1
            frame_idx = 0
            word_samples = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % step != 0:
                    continue

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                features, has_hands = extract_hand_features(results)

                if has_hands:
                    word_samples.append(features)

            cap.release()

            if not word_samples:
                print(f"Skipping {word} (no hands detected in video)")
                continue

            # Add extracted frames
            class_id = len(labels)
            labels.append(word)
            for s in word_samples:
                X.append(s)
                y.append(class_id)
                # Augmentation with slight noise and rotation
                noise = np.random.normal(0, 0.01, s.shape)
                X.append(s + noise)
                y.append(class_id)

            print(f"[{idx+1}/{len(TARGET_50)}] {word}: extracted {len(word_samples)*2} frame samples", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\nTotal per-frame samples: {len(X)}, Feature dimensions: {X.shape[1]}")
    print(f"Total sign classes: {len(labels)}")

    np.savez_compressed("handshape_dataset.npz", X=X, y=y)
    with open("handshape_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print("Saved handshape_dataset.npz and handshape_labels.json successfully!")

if __name__ == "__main__":
    main()
