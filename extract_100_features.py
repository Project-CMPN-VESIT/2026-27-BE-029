import os
import cv2
import json
import time
import numpy as np
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor

def compute_hand_geometry(landmarks, wrist, span):
    # 5 finger curl ratios (tip to MCP distance / span)
    curls = [
        np.linalg.norm(landmarks[4] - landmarks[2]) / span,   # Thumb
        np.linalg.norm(landmarks[8] - landmarks[5]) / span,   # Index
        np.linalg.norm(landmarks[12] - landmarks[9]) / span,  # Middle
        np.linalg.norm(landmarks[16] - landmarks[13]) / span, # Ring
        np.linalg.norm(landmarks[20] - landmarks[17]) / span  # Pinky
    ]
    # 4 inter-fingertip spread distances
    spreads = [
        np.linalg.norm(landmarks[4] - landmarks[8]) / span,   # Thumb-Index
        np.linalg.norm(landmarks[8] - landmarks[12]) / span,  # Index-Middle
        np.linalg.norm(landmarks[12] - landmarks[16]) / span, # Middle-Ring
        np.linalg.norm(landmarks[16] - landmarks[20]) / span  # Ring-Pinky
    ]
    # Palm normal vector
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

    # Left hand
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

    # Right hand
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

    # Base 135 features
    base_135 = np.concatenate([
        lh_features,
        rh_features,
        [lh_present, rh_present],
        lh_pos,
        rh_pos,
        [inter_dist]
    ])

    # Enhanced 159 features
    feat_159 = np.concatenate([base_135, lh_extra, rh_extra])

    has_hands = (lh_present > 0 or rh_present > 0)
    return feat_159, has_hands

def process_single_video(args):
    vpath, label, label_id = args
    cap = cv2.VideoCapture(vpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 3 if total_frames > 200 else 2

    mp_holistic = mp.solutions.holistic
    feats = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
        model_complexity=0
    ) as holistic:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                frame_small = cv2.resize(frame, (480, 270))
                rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                feat, has_hands = extract_features(results)
                if has_hands:
                    feats.append(feat)
            frame_idx += 1

    cap.release()
    print(f"Finished {label}: {len(feats)} frames", flush=True)
    return label_id, feats

def main():
    video_dir = "words_100_videos"
    if not os.path.exists(video_dir):
        print(f"Error: {video_dir} not found.", flush=True)
        return

    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    print(f"Starting 4-worker parallel feature extraction for {len(video_files)} word videos...", flush=True)

    labels = [os.path.splitext(f)[0] for f in video_files]
    label_to_id = {label: i for i, label in enumerate(labels)}

    with open("words_100_labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    tasks = []
    for vf in video_files:
        label = os.path.splitext(vf)[0]
        vpath = os.path.join(video_dir, vf)
        tasks.append((vpath, label, label_to_id[label]))

    start_time = time.time()
    all_features = []
    all_labels = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_single_video, tasks)
        for label_id, feats in results:
            for f in feats:
                all_features.append(f)
                all_labels.append(label_id)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    elapsed = time.time() - start_time
    print(f"\nParallel feature extraction complete in {elapsed:.1f}s!", flush=True)
    print(f"Total dataset shape: X={X.shape}, y={y.shape}", flush=True)
    print(f"Saved to words_100_dataset.npz and words_100_labels.json", flush=True)

    np.savez_compressed("words_100_dataset.npz", X=X, y=y)

if __name__ == "__main__":
    main()
