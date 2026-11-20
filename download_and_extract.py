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

def extract_landmarks_from_results(results):
    pose_indices = [0, 11, 12, 13, 14, 15, 16]
    has_hand = False
    
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x,
                          results.pose_landmarks.landmark[i].y,
                          results.pose_landmarks.landmark[i].z] for i in pose_indices]).flatten()
        ref_x = (pose[3] + pose[6]) / 2.0
        ref_y = (pose[4] + pose[7]) / 2.0
        scale = np.sqrt((pose[3] - pose[6])**2 + (pose[4] - pose[7])**2)
        if scale < 0.05:
            scale = 0.2
    else:
        pose = np.zeros(len(pose_indices) * 3)
        ref_x, ref_y, scale = 0.5, 0.5, 0.2

    if results.left_hand_landmarks:
        has_hand = True
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        has_hand = True
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    raw = np.concatenate([pose, lh, rh])
    normed = raw.copy()
    for i in range(0, len(normed), 3):
        if normed[i] != 0 or normed[i+1] != 0:
            normed[i] = (normed[i] - ref_x) / scale
            normed[i+1] = (normed[i+1] - ref_y) / scale
            normed[i+2] = normed[i+2] / scale

    return normed, has_hand

def process_video(video_path, holistic, target_frames=30):
    cap = cv2.VideoCapture(video_path)
    active_keypoints = []
    all_keypoints = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = 2 if fps > 40 else 1
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        keypoints, has_hand = extract_landmarks_from_results(results)
        
        all_keypoints.append(keypoints)
        if has_hand:
            active_keypoints.append(keypoints)

    cap.release()

    # Prioritize active frames where hands were actually performing the sign
    frames_to_use = active_keypoints if len(active_keypoints) >= 10 else all_keypoints
    if not frames_to_use:
        return None

    frames_np = np.array(frames_to_use)
    T = len(frames_np)
    indices = np.linspace(0, T - 1, target_frames).astype(int)
    sampled = frames_np[indices]
    return sampled

def augment_sequence(seq, num_aug=20):
    augmented = [seq]
    T, D = seq.shape
    for _ in range(num_aug - 1):
        aug = seq.copy()
        noise = np.random.normal(0, 0.015, aug.shape)
        aug = aug + noise
        scale_factor = np.random.uniform(0.92, 1.08)
        aug = aug * scale_factor
        shift = np.random.uniform(-0.03, 0.03, (1, D))
        aug = aug + shift
        augmented.append(aug)
    return augmented

def main():
    print("--- Extracting ACTIVE Hand Movement Landmarks ---")
    mp_holistic = mp.solutions.holistic

    X_list = []
    y_list = []
    final_labels = []

    with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
        for idx, word in enumerate(TARGET_50):
            video_path = os.path.join("videos", f"{word}.mp4")
            if not os.path.exists(video_path):
                continue

            print(f"[{idx+1}/{len(TARGET_50)}] Processing active gesture frames for {word}...", flush=True)
            seq = process_video(video_path, holistic, target_frames=30)
            if seq is None:
                continue

            aug_samples = augment_sequence(seq, num_aug=20)
            for s in aug_samples:
                X_list.append(s)
                y_list.append(len(final_labels))

            final_labels.append(word)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"\nActive dataset created! Total sequences: {len(X)}, Classes: {len(final_labels)}", flush=True)
    np.savez_compressed("dataset.npz", X=X, y=y)
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(final_labels, f, ensure_ascii=False, indent=2)
    print("dataset.npz updated successfully!", flush=True)

if __name__ == "__main__":
    main()
