"""Inference helpers for the trained intent-classification model."""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing import MAX_SEQUENCE_LENGTH, normalize_text

FALLBACK_RESPONSE = (
    "I'm not confident I understood that. Try asking about admissions, fees, or campus life. "
    "I can also help with courses, exams, hostel, placements, library, scholarships, attendance, transport, and student activities."
)
STOP_WORDS = {
    "a", "an", "and", "are", "about", "can", "could", "do", "does", "for", "how", "i", "is", "me", "my",
    "of", "on", "please", "the", "to", "what", "when", "where", "who", "will", "with", "you", "your",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_responses(dataset_path: Path | None = None) -> dict[str, list[str]]:
    dataset_path = dataset_path or project_root() / "dataset" / "intents.json"
    with open(dataset_path, encoding="utf-8") as handle:
        return {item["intent"]: item["responses"] for item in json.load(handle)["intents"]}


def load_artifacts(model_dir: Path | None = None):
    model_dir = model_dir or project_root() / "model"
    required = [model_dir / "chatbot_model.keras", model_dir / "tokenizer.pkl", model_dir / "label_encoder.pkl"]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing trained artifact(s): " + ", ".join(missing) + ". Run: python train.py")
    classifier = model_dir / "intent_classifier.pkl"
    vectorizer = model_dir / "vectorizer.pkl"
    if classifier.exists() and vectorizer.exists():
        return joblib.load(classifier), joblib.load(vectorizer), joblib.load(required[2])
    return load_model(required[0]), joblib.load(required[1]), joblib.load(required[2])


def predict_intent(text: str, model, tokenizer, label_encoder, threshold: float = 0.50) -> tuple[str | None, float]:
    top_intents = predict_top_intents(text, model, tokenizer, label_encoder)
    if not top_intents:
        return None, 0.0
    intent, confidence = top_intents[0]
    if confidence < threshold:
        return None, confidence
    return intent, confidence


def predict_top_intents(text: str, model, tokenizer, label_encoder, limit: int = 3) -> list[tuple[str, float]]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if hasattr(tokenizer, "transform"):
        probabilities = model.predict_proba(tokenizer.transform([cleaned]))[0]
        indices = np.argsort(probabilities)[::-1][:limit]
        return [
            (str(label_encoder.inverse_transform([int(index)])[0]), float(probabilities[index]))
            for index in indices
        ]
    sequence = tokenizer.texts_to_sequences([cleaned])
    oov_index = tokenizer.word_index.get(tokenizer.oov_token)
    tokens = sequence[0]
    if not tokens or (oov_index and sum(token == oov_index for token in tokens) / len(tokens) >= 0.40):
        return []
    content_words = [word for word in cleaned.split() if word not in STOP_WORDS]
    known_content = [word for word in content_words if tokenizer.word_index.get(word) not in (None, oov_index)]
    if content_words and len(known_content) / len(content_words) <= 0.50:
        return []
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    indices = np.argsort(probabilities)[::-1][:limit]
    return [
        (str(label_encoder.inverse_transform([int(index)])[0]), float(probabilities[index]))
        for index in indices
    ]


def reply_for(text: str, model, tokenizer, label_encoder, responses: dict[str, list[str]], threshold: float = 0.50):
    intent, confidence = predict_intent(text, model, tokenizer, label_encoder, threshold)
    response = random.choice(responses[intent]) if intent else FALLBACK_RESPONSE
    return intent, confidence, response
