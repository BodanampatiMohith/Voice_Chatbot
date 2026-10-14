---
title: Voice-Enabled University Chatbot
emoji: 🎓
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.64.0
app_file: app.py
pinned: false
---

# Campus Voice Assistant

A fully offline, voice-enabled university information chatbot built with Streamlit, TensorFlow/Keras, and Vosk. Ask questions by voice or text — no internet connection, API key, or cloud speech service required after the first setup.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Local Setup](#local-setup)
- [Training the Model](#training-the-model)
- [Running the App](#running-the-app)
- [Testing](#testing)
- [Deployment](#deployment)
- [Evaluation](#evaluation)
- [Limitations](#limitations)

---

## Features

- **Offline voice recognition** — records from the browser microphone, normalizes audio to 16 kHz mono PCM, and transcribes locally using Vosk (no cloud STT API).
- **18 university intents** — admissions, fees, hostel, courses, exams, faculty, library, placements, campus, timetable, transport, scholarships, attendance, student activities, contact, greeting, goodbye, thanks.
- **Dual-model inference** — production uses TF-IDF + Logistic Regression; a trained Bi-LSTM is also saved for comparison.
- **Confidence threshold** — adjustable slider; uncertain predictions fall back to a helpful suggestion message.
- **Top-3 intent scores** — shown alongside the primary prediction for transparency.
- **Full conversation history** with a clear-chat control.
- **Text input fallback** — works without a microphone.

---

## Project Structure

```
Voice_Chatbot_AIR/
├── app.py                     # Streamlit UI entry point
├── train.py                   # Training pipeline (Bi-LSTM + TF-IDF baseline)
├── test_chatbot.py            # Smoke tests (run after training)
├── requirements.txt           # Python dependencies
├── runtime.txt                # Python version pin for cloud deployment
│
├── dataset/
│   └── intents.json           # 18 intents × 40 patterns + 3 responses each
│
├── model/
│   ├── chatbot_model.keras    # Trained Bi-LSTM model
│   ├── intent_classifier.pkl  # Production TF-IDF + Logistic Regression
│   ├── vectorizer.pkl         # TF-IDF vectorizer (production)
│   ├── tokenizer.pkl          # Keras tokenizer (Bi-LSTM)
│   ├── label_encoder.pkl      # Sklearn LabelEncoder
│   ├── training_metrics.json  # Held-out test scores + classification report
│   ├── confusion_matrix.json  # Raw confusion matrix data
│   └── training_history.json  # Per-epoch loss/accuracy history
│
├── src/
│   ├── __init__.py
│   ├── chatbot.py             # Inference helpers (load, predict, reply)
│   ├── preprocessing.py       # Shared text normalization
│   └── speech_recognition_utils.py  # Audio conversion + Vosk transcription
│
├── report/
│   ├── training_curves.png    # Loss and accuracy vs epoch
│   ├── confusion_matrix.png   # Per-intent confusion matrix heatmap
│   └── model_comparison.png   # Bi-LSTM vs TF-IDF baseline bar chart
│
└── .streamlit/
    └── config.toml            # Streamlit theme configuration
```

---

## Architecture

### Inference pipeline

```
Browser mic
    │
    ▼
st.audio_input  ──────────────────────────────────────────┐
    │                                                      │
    ▼                                                   Text input
convert_to_16khz_mono_wav()                               │
    │                                                      │
    ▼                                                      │
KaldiRecognizer (Vosk offline ASR)                         │
    │                                                      │
    ▼                                                      │
recognized text  ◄─────────────────────────────────────────┘
    │
    ▼
normalize_text()   ← lowercase, strip punctuation, collapse whitespace
    │
    ▼
TF-IDF vectorizer  →  Logistic Regression  →  class probabilities
    │
    ▼
confidence ≥ threshold?
    ├─ YES → return intent + random response from intents.json
    └─ NO  → return fallback suggestion message
```

### Model architecture (Bi-LSTM — saved for comparison)

```
Input (sequence length 20)
    │
Embedding (vocab × 128)
    │
SpatialDropout1D (0.20)
    │
Bidirectional LSTM (96 units, return_sequences=True, dropout=0.15)
    │
    ├── GlobalMaxPooling1D ──┐
    └── GlobalAveragePooling1D ─► Concatenate (192-d)
                                    │
                                Dense (96, ReLU)
                                    │
                                Dropout (0.35)
                                    │
                                Dense (18, Softmax)
```

Training uses label smoothing (0.08), early stopping (patience=9), and ReduceLROnPlateau. The training partition is augmented with synonym replacement and function-word deletion.

---

## Dataset

`dataset/intents.json` contains 18 intent classes, each with:
- **40 distinct patterns** (utterances)
- **3 response variants** (randomly selected at inference)

| Intent | Topic |
|---|---|
| `admission` | Application, counselling, eligibility |
| `fees` | Tuition, semester charges, online payment |
| `courses` | Available programmes, durations |
| `faculty` | Teaching staff, profiles |
| `library` | Borrowing, renewal, timings |
| `hostel` | Rooms, mess menu, facilities |
| `placements` | Campus recruitment, mock interviews |
| `campus` | Location, map, facilities |
| `timetable` | Class schedules, lecture timings |
| `examinations` | Exam dates, hall tickets, results |
| `contact` | Email, phone, official channels |
| `scholarships` | Availability, renewal, eligibility |
| `attendance` | Minimum requirement, checking attendance |
| `transport` | Bus routes, bus pass |
| `student_activities` | Clubs, college fest, events |
| `greeting` | Hello, hi, namaste |
| `goodbye` | Bye, signing off |
| `thanks` | Thank you, acknowledgement |

---

## Local Setup

> Requires **Python 3.11+** (tested on 3.14). See `runtime.txt`.

```powershell
# 1. Create and activate a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Train the model
python train.py

# 4. Run smoke tests
python test_chatbot.py

# 5. Start the app
python -m streamlit run app.py
```

The Vosk speech model (`vosk-model-small-en-us-0.15`, ~40 MB) is downloaded automatically to `model/vosk/` on first use. After that, all transcription is fully offline.

---

## Training the Model

`train.py` performs a full training run and saves all artifacts to `model/`:

- Stratified 70 / 15 / 15 train / validation / test split
- Augments only the training partition (synonym swap, function-word deletion)
- Fits the Keras tokenizer on training data only
- Trains the Bi-LSTM with early stopping and learning-rate reduction
- Trains a TF-IDF + Logistic Regression baseline for comparison
- Saves the production classifier (`intent_classifier.pkl` + `vectorizer.pkl`)
- Saves the Bi-LSTM (`chatbot_model.keras` + `tokenizer.pkl`)
- Saves held-out metrics, confusion matrix, and training history to `model/`
- Saves training curves, confusion-matrix heatmap, and model-comparison plots to `report/`

Re-run `python train.py` whenever `dataset/intents.json` is updated.

---

## Running the App

```powershell
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

- Use the microphone widget to record a question, then click **Transcribe and ask**.
- Or type directly in the text box and click **Ask chatbot**.
- Adjust the **Confidence threshold** slider to control how certain the model must be before giving a direct answer.

---

## Testing

```powershell
python test_chatbot.py
```

Runs 36 labelled intent cases (2 per intent), two out-of-domain rejection checks, an empty-input check, and two audio edge-case checks (empty bytes, silent WAV). All cases must pass before deployment.

---

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub (include trained model artifacts; exclude `.venv/` and `model/vosk/`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the main file to `app.py`.
3. The cloud environment installs dependencies from `requirements.txt` automatically.
4. Vosk downloads its model on first cold start.

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) with the **Streamlit** SDK.
2. Push this repository to the Space's Git remote.
3. The YAML front-matter at the top of this file registers the app entry point and SDK version automatically.

```powershell
git add app.py train.py test_chatbot.py src dataset report model requirements.txt runtime.txt README.md .gitignore .streamlit
git commit -m "Deploy offline voice-enabled university chatbot"
git push
```

---

## Evaluation

All metrics are generated by `train.py` against the untouched 15% held-out test split and saved to `model/training_metrics.json`. Report plots are saved to `report/`.

| Metric | Bi-LSTM | TF-IDF + LR |
|---|---|---|
| Accuracy | see `training_metrics.json` | see `training_metrics.json` |
| Weighted F1 | see `training_metrics.json` | see `training_metrics.json` |

No numbers are hardcoded here — regenerate with `python train.py` after any dataset change to get fresh, honest figures.

---

## Limitations

- **Closed-set classification** — any question outside the 18 defined intents triggers the fallback response.
- **Compact dataset** — 40 utterances per intent; real-world robustness depends on broader coverage.
- **US English ASR** — Vosk's small English model is tuned for US accent; other accents may have lower transcription accuracy.
- **Not an official source** — responses are illustrative. Always verify against current university notices.
