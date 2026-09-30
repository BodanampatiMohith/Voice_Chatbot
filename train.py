"""Train and save a genuine Bi-LSTM intent classifier."""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Concatenate, Dense, Dropout, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, Input, LSTM, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.preprocessing import MAX_SEQUENCE_LENGTH, normalize_text

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset" / "intents.json"
MODEL_DIR = ROOT / "model"
SEED = 42
SYNONYMS = {
    "college": "university", "university": "college", "fees": "tuition",
    "class": "lecture", "lectures": "classes", "professor": "teacher",
    "bus": "transport", "hostel": "residence", "scholarship": "financial",
}


def load_training_data():
    with open(DATASET, encoding="utf-8") as handle:
        intents = json.load(handle)["intents"]
    texts, labels = [], []
    for item in intents:
        for pattern in item["patterns"]:
            texts.append(normalize_text(pattern))
            labels.append(item["intent"])
    return texts, labels


def build_model(vocabulary_size: int, classes: int) -> Model:
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    sequence = Embedding(vocabulary_size, 128)(inputs)
    sequence = SpatialDropout1D(0.20)(sequence)
    sequence = Bidirectional(LSTM(96, return_sequences=True, dropout=0.15))(sequence)
    pooled = Concatenate()([
        GlobalMaxPooling1D()(sequence),
        GlobalAveragePooling1D()(sequence),
    ])
    features = Dense(96, activation="relu")(pooled)
    features = Dropout(0.35)(features)
    outputs = Dense(classes, activation="softmax")(features)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy"],
    )
    return model


def augment_training_text(text: str, rng: random.Random) -> str:
    """Apply one light, label-preserving word edit to a training utterance."""
    words = text.split()
    synonym_positions = [index for index, word in enumerate(words) if word in SYNONYMS]
    if synonym_positions and rng.random() < 0.45:
        index = rng.choice(synonym_positions)
        words[index] = SYNONYMS[words[index]]
    elif len(words) > 4 and rng.random() < 0.65:
        removable = [index for index, word in enumerate(words) if word in {"a", "the", "my", "is", "do", "can", "please"}]
        if removable:
            words.pop(rng.choice(removable))
        else:
            words.append("please")
    else:
        words.append("please")
    return " ".join(words)


def weighted_scores(expected, predicted) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, average="weighted", zero_division=0
    )
    return {"weighted_precision": float(precision), "weighted_recall": float(recall), "weighted_f1": float(f1)}


def save_plots(history, labels, matrix, neural_scores, baseline_scores) -> None:
    report_dir = ROOT / "report"
    report_dir.mkdir(exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.history["loss"], label="Training")
    axes[0].plot(history.history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Categorical cross-entropy")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Training")
    axes[1].plot(history.history["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(report_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
             xticklabels=labels, yticklabels=labels, xlabel="Predicted intent", ylabel="True intent",
             title="Bi-LSTM test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    score_names = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1"]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    y_positions = np.arange(len(score_names))
    axis.barh(y_positions + 0.18, [neural_scores[name] for name in score_names], height=0.34, label="Bi-LSTM")
    axis.barh(y_positions - 0.18, [baseline_scores[name] for name in score_names], height=0.34, label="TF-IDF + Logistic Regression")
    axis.set(yticks=y_positions, yticklabels=[name.replace("weighted_", "weighted ").title() for name in score_names],
             xlim=(0, 1), xlabel="Held-out test score", title="Model comparison")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(report_dir / "model_comparison.png", dpi=160)
    plt.close(figure)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    texts, labels = load_training_data()
    encoder = LabelEncoder()
    train_texts, holdout_texts, y_train_names, holdout_names = train_test_split(
        texts, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_texts, test_texts, y_val_names, y_test_names = train_test_split(
        holdout_texts, holdout_names, test_size=0.50, random_state=SEED, stratify=holdout_names
    )
    encoder.fit(y_train_names)
    y_train = encoder.transform(y_train_names)
    y_val = encoder.transform(y_val_names)
    y_test = encoder.transform(y_test_names)

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    rng = random.Random(SEED)
    augmented_texts = train_texts + [augment_training_text(text, rng) for text in train_texts]
    augmented_labels = list(y_train) + list(y_train)
    x_train = pad_sequences(tokenizer.texts_to_sequences(augmented_texts), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    x_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    y_train_onehot = tf.keras.utils.to_categorical(augmented_labels, num_classes=len(encoder.classes_))
    y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=len(encoder.classes_))

    model = build_model(len(tokenizer.word_index) + 1, len(encoder.classes_))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=9, min_delta=0.001, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001, verbose=1),
    ]
    history = model.fit(x_train, y_train_onehot, validation_data=(x_val, y_val_onehot),
                        epochs=100, batch_size=32, callbacks=callbacks, verbose=2)
    training_loss, training_accuracy = model.evaluate(
        pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"),
        tf.keras.utils.to_categorical(y_train, num_classes=len(encoder.classes_)), verbose=0,
    )
    test_probabilities = model.predict(x_test, verbose=0)
    predictions = test_probabilities.argmax(axis=1)
    neural_scores = {"accuracy": float(np.mean(predictions == y_test)), **weighted_scores(y_test, predictions)}
    baseline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    x_baseline_train = baseline_vectorizer.fit_transform(train_texts)
    baseline = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    baseline.fit(x_baseline_train, y_train)
    baseline_predictions = baseline.predict(baseline_vectorizer.transform(test_texts))
    baseline_scores = {"accuracy": float(np.mean(baseline_predictions == y_test)), **weighted_scores(y_test, baseline_predictions)}

    production_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    production_features = production_vectorizer.fit_transform(texts)
    production_baseline = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=SEED)
    production_baseline.fit(production_features, encoder.transform(labels))

    matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(encoder.classes_)))
    report = classification_report(y_test, predictions, labels=np.arange(len(encoder.classes_)),
                                   target_names=encoder.classes_, output_dict=True, zero_division=0)
    baseline_report = classification_report(y_test, baseline_predictions, labels=np.arange(len(encoder.classes_)),
                                            target_names=encoder.classes_, output_dict=True, zero_division=0)
    best_loss_epoch = int(np.argmin(history.history["val_loss"])) + 1
    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    validation_probabilities = model.predict(x_val, verbose=0)
    validation_accuracy = float(np.mean(validation_probabilities.argmax(axis=1) == y_val))
    print(f"Samples: {len(texts)} across {len(encoder.classes_)} intents; stratified split train/validation/test: {len(train_texts)}/{len(val_texts)}/{len(test_texts)}")
    print(f"Train-only tokenizer vocabulary: {len(tokenizer.word_index)} words; augmented training examples: {len(augmented_texts)}")
    print(f"Bi-LSTM training loss/accuracy: {training_loss:.4f}/{training_accuracy:.4f}")
    print(f"Bi-LSTM validation accuracy: {validation_accuracy:.4f}")
    print("Bi-LSTM held-out test metrics:", json.dumps(neural_scores, indent=2))
    print("TF-IDF + Logistic Regression held-out test metrics:", json.dumps(baseline_scores, indent=2))
    print("Bi-LSTM classification report:\n", classification_report(y_test, predictions, labels=np.arange(len(encoder.classes_)), target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):\n", matrix)

    metrics = {
        "dataset_samples": len(texts), "intent_classes": len(encoder.classes_), "samples_per_intent": 40,
        "split": {"method": "stratified", "train": len(train_texts), "validation": len(val_texts), "test": len(test_texts), "ratios": "70/15/15"},
        "tokenizer_vocabulary_size": len(tokenizer.word_index),
        "augmentation": "one light synonym, function-word deletion, or please insertion per training utterance; validation/test unchanged",
        "architecture": "Embedding(128) -> SpatialDropout1D(0.20) -> BiLSTM(96, return_sequences=True) -> GlobalMaxPool + GlobalAveragePool -> Dense(96, ReLU) -> Dropout(0.35) -> Softmax; categorical cross-entropy label smoothing 0.08",
        "training_loss": float(training_loss), "training_accuracy": float(training_accuracy),
        "validation_accuracy": validation_accuracy, "epochs_completed": len(history.history["loss"]),
        "best_validation_loss": float(np.min(history.history["val_loss"])), "best_validation_loss_epoch": best_loss_epoch,
        "neural_test": neural_scores, "neural_classification_report": report,
        "baseline_test": baseline_scores, "baseline_classification_report": baseline_report,
    }
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODEL_DIR / "confusion_matrix.json").write_text(
        json.dumps({"labels": encoder.classes_.tolist(), "matrix": matrix.tolist()}, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "training_history.json").write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    save_plots(history, encoder.classes_, matrix, neural_scores, baseline_scores)
    model.save(MODEL_DIR / "chatbot_model.keras")
    joblib.dump(tokenizer, MODEL_DIR / "tokenizer.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(production_vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(production_baseline, MODEL_DIR / "intent_classifier.pkl")
    print("Saved trained model, train-only tokenizer, label encoder, held-out metrics, and report plots.")


if __name__ == "__main__":
    main()
