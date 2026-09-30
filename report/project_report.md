# Voice-Enabled Chatbot Using Speech Recognition and Deep Learning

## Abstract

This project is a Streamlit university-information assistant. Browser-recorded audio is converted to 16 kHz mono PCM WAV and transcribed locally with Vosk. A TensorFlow/Keras Bi-LSTM predicts one of 18 intents and selects a response variant from an original dataset. Typed input provides an alternative to the microphone; low-confidence requests receive a scope-aware fallback.

## Dataset and intent boundaries

`dataset/intents.json` contains 18 intents with exactly 40 distinct utterances per intent (720 utterances total) and three response variants for each class. Utterances use conversational English and Indian-English phrasing. The intent boundaries are embedded in the dataset: fees covers general tuition and academic charges; hostel covers residence, rooms, mess, and accommodation costs; timetable covers class schedules; examinations covers exam administration and results; transport covers university buses and routes. Separate classes cover admissions, courses, faculty, library, placements, campus, scholarships, attendance, and student activities, alongside greeting, goodbye, and thanks.

Preprocessing lowercases text, removes punctuation outside letters/numbers/apostrophes, tokenizes, and pads sequences to 20 tokens. The tokenizer is fitted on training text only. One light synonym substitution, function-word deletion, or polite-word insertion is generated per training utterance; validation and test examples remain unchanged.

## Offline speech recognition methodology

The app uses Streamlit `st.audio_input`. SoundFile reads supported browser audio, SciPy resamples when needed, and the app writes mono, 16 kHz, 16-bit PCM WAV in memory. Vosk's `vosk-model-small-en-us-0.15` performs local recognition. The model is downloaded on first voice use and cached in `model/vosk/`, which is excluded from Git. Empty, silent, and invalid audio produce readable messages. After the one-time model download, audio is not sent to an online recognizer.

## Model and training methodology

`train.py` creates a reproducible stratified 70/15/15 train/validation/test split. The specified architecture is:

`Embedding(128) -> SpatialDropout1D(0.20) -> Bidirectional LSTM(96, return_sequences=True) -> concatenate(GlobalMaxPooling1D, GlobalAveragePooling1D) -> Dense(96, ReLU) -> Dropout(0.35) -> Dense(18, Softmax)`.

Adam uses a 0.001 initial learning rate and categorical cross-entropy with 0.08 label smoothing. EarlyStopping restores the best validation-loss weights; ReduceLROnPlateau reduces the learning rate when validation loss stalls. A separate TF-IDF (unigrams and bigrams) plus class-balanced Logistic Regression baseline is evaluated on the same untouched test split.

## Training setup and results

The intended deployment environment is Python 3.11 with versions pinned in `requirements.txt`. A successful run saves the trained `.keras` model, tokenizer, label encoder, training metrics, test confusion matrix, and epoch history under `model/`; training curves, confusion matrix, and baseline-comparison graphics are saved in this directory.

**Results are pending.** In this Windows workspace, TensorFlow training could not start: Windows Application Control blocked TensorFlow's native runtime DLL before the script entered training. The installed Python 3.12 interpreter has no TensorFlow package. Therefore, no current held-out metrics are reported, and the prior metrics/artifacts in `model/` are from the superseded dataset and must not be treated as results for this version. Run `python train.py` in an allowed Python 3.11 environment, then update this section from `model/training_metrics.json`; do not estimate or invent scores.

Generated evaluation artifacts:

- `model/training_metrics.json`: neural and baseline accuracy, weighted precision/recall/F1, and per-class classification reports.
- `model/confusion_matrix.json`: neural test confusion matrix with label order.
- `model/training_history.json`: training/validation losses and accuracies by epoch.
- `report/training_curves.png`, `report/confusion_matrix.png`, and `report/model_comparison.png`: generated plots.

## Tests

`test_chatbot.py` checks two utterances for each of the 18 intents, two unrelated questions, empty text, empty audio, and silent audio. It expects newly trained artifacts; the existing artifacts in this workspace do not represent the expanded dataset, and the test run also depends on a working TensorFlow runtime.

## Deployment

The README starts with the Hugging Face Spaces Streamlit YAML metadata and gives the steps to create a Space, regenerate model artifacts, commit those artifacts, and push the project. The Space downloads Vosk on first use. No deployment URL exists yet.

## Limitations

This is an educational intent classifier, not an official university source or open-domain language model. Unfamiliar language may be misclassified, and the model's confidence is not a guarantee of factual correctness. Vosk's small US-English model may perform less well with diverse accents or noisy recordings. Verify current dates, charges, eligibility, and policies with official university notices.
