# VANI-ISL: Real-Time Indian Sign Language Recognizer

A fast, low-latency, real-time Indian Sign Language (ISL) recognition system. It translates 220 hand signs and continuous sentences directly from a standard laptop webcam into text and natural English speech.

---

## 🚀 How to Run

### Option 1: 1-Click Launch (Easiest)
- Double-click **`Start_ISL_Recognizer.bat`** on your Desktop, OR
- Double-click **`run_camera.bat`** inside this folder.

### Option 2: Command Prompt / Terminal
Open Command Prompt and run:
```cmd
cd /d "C:\Users\USER\Desktop\ISL_RealTime_Translator"
run_camera.bat
```

---

## 🎮 Controls Cheat Sheet

| Key | Action | Description |
| :---: | :--- | :--- |
| **`s`** | **Speak Sentence** | Reads the constructed sentence out loud in clear English |
| **`b`** or **`Backspace`** | **Undo / Delete** | Removes the last added word from your sentence |
| **`r`** | **Replace Word** | Replaces the last word with the sign currently held |
| **`Space`** | **Manual Add** | Manually adds the current sign to the sentence |
| **`a`** | **Toggle Auto-Add** | Switch between Auto-Add (hands-free dwell) and Manual (Space) |
| **`c`** | **Clear** | Clears the entire current sentence |
| **`m`** or **`Tab`** | **Switch Mode** | Cycle: 50 Core Words $\leftrightarrow$ 100 Extended $\leftrightarrow$ 70 Sentences $\leftrightarrow$ All 220 |
| **`v`** | **Mute / Unmute** | Toggle voice speech synthesizer audio |
| **`g`** | **Guide Window** | Toggle demonstrator reference video in the corner |
| **`n` / `p`** | **Next / Prev** | Switch reference guide video to next / previous sign |
| **`k`** | **Save Snapshot** | Saves current camera frame image to disk |
| **`q`** or **`Esc`** | **Quit** | Exit the application |

---

## 🤲 Hands-Free Signing (Two-Handed Mode)
In Indian Sign Language, many gestures require **both hands**, making it difficult to reach for a keyboard or mouse.
1. Make your sign and hold it steady in view of the webcam for ~0.4s.
2. A progress indicator appears: `Hold: Hello (50%)` $\to$ `LOCKED: Hello (100%)`.
3. **Lower or release your hands:** The sign is automatically appended to your sentence!

---

## 📂 Project Structure

- **`camera_recognizer.py`**: Main OpenCV and MediaPipe recognition application.
- **`run_camera.bat`**: Startup script configuring the Python environment.
- **Models (`.h5` / `.tflite`):**
  - `handshape_model.h5`: 50 Core Signs neural classifier (96.65% accuracy).
  - `words_100_model.h5`: 100 Extended Vocabulary classifier (81.29% accuracy).
  - `sentence_model.h5`: 70 Full Continuous Sentences classifier (78.11% accuracy).
- **Labels & Hints (`.json`):**
  - `handshape_labels.json`, `sign_hints.json`
  - `words_100_labels.json`, `words_100_hints.json`
  - `sentence_labels.json`, `sentence_hints.json`
  - `catalog.json`
- **Reference Video Guides:**
  - `videos/`: Reference demonstration clips for 50 core words.
  - `words_100_videos/`: Reference demonstration clips for 100 extended signs.
  - `sentence_videos/`: Reference demonstration clips for 70 sentences.
- **Datasets (`.npz`):**
  - `dataset.npz`, `handshape_dataset.npz`, `words_100_dataset.npz`, `sentence_dataset.npz`.
- **Training & Feature Extraction Scripts:**
  - `extract_hand_shapes.py`, `extract_100_features.py`, `extract_sentence_features.py`
  - `train_model.py`, `train_handshape_model.py`, `train_100_model.py`, `train_sentence_model.py`

---

