import os
import json
import cv2
import numpy as np
import mediapipe as mp

def extract_hand_features(results):
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
    if results.left_hand_landmarks:
        lh_present = 1.0
        lm = results.left_hand_landmarks.landmark
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
        lh_pos = (wrist - np.array([ref_x, ref_y, 0.0])) / sh_dist
        
        rel_points = []
        for p in lm:
            rel_points.append([p.x - wrist[0], p.y - wrist[1], p.z - wrist[2]])
        rel_points = np.array(rel_points)
        span = np.max(np.linalg.norm(rel_points, axis=1))
        span = max(span, 0.02)
        lh_features = (rel_points / span).flatten()

    rh_features = np.zeros(21 * 3)
    rh_present = 0.0
    rh_pos = np.zeros(3)
    if results.right_hand_landmarks:
        rh_present = 1.0
        lm = results.right_hand_landmarks.landmark
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
        rh_pos = (wrist - np.array([ref_x, ref_y, 0.0])) / sh_dist

        rel_points = []
        for p in lm:
            rel_points.append([p.x - wrist[0], p.y - wrist[1], p.z - wrist[2]])
        rel_points = np.array(rel_points)
        span = np.max(np.linalg.norm(rel_points, axis=1))
        span = max(span, 0.02)
        rh_features = (rel_points / span).flatten()

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
    video_dir = "sentence_videos"
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    print(f"Fast feature extraction across {len(video_files)} sentence videos...", flush=True)

    mp_holistic = mp.solutions.holistic

    X = []
    y = []
    labels = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
        model_complexity=1
    ) as holistic:
        for idx, filename in enumerate(video_files):
            sentence_name = filename[:-4]
            video_path = os.path.join(video_dir, filename)

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Fast sampling step:
            # - For very long videos (>400 frames), step every 4 frames
            # - For typical videos (70-400 frames), step every 3 frames
            step = 4 if total_frames > 400 else 3

            frame_idx = 0
            sentence_samples = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % step != 0:
                    continue

                # Fast processing resolution
                frame_resized = cv2.resize(frame, (480, 270))
                image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                features, has_hands = extract_hand_features(results)

                if has_hands:
                    sentence_samples.append(features)

            cap.release()

            if not sentence_samples:
                print(f"[{idx+1}/{len(video_files)}] Skipping {sentence_name} (no hands)", flush=True)
                continue

            class_id = len(labels)
            labels.append(sentence_name)

            for s in sentence_samples:
                X.append(s)
                y.append(class_id)
                # Augmentation with slight variation
                noise = np.random.normal(0, 0.012, s.shape)
                X.append(s + noise)
                y.append(class_id)

            print(f"[{idx+1}/{len(video_files)}] {sentence_name}: {len(sentence_samples)*2} samples", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\nTotal sentence dataset samples: {len(X)}, Feature dimensions: {X.shape[1]}", flush=True)
    print(f"Total sentence classes: {len(labels)}", flush=True)

    np.savez_compressed("sentence_dataset.npz", X=X, y=y)
    with open("sentence_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("Saved sentence_dataset.npz and sentence_labels.json successfully!", flush=True)

if __name__ == "__main__":
    main()
