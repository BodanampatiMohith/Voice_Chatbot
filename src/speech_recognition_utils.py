"""Offline Vosk speech recognition and audio normalization helpers."""
from __future__ import annotations

import io
import json
import math
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from vosk import KaldiRecognizer, Model


MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
SAMPLE_RATE = 16_000


def ensure_vosk_model(model_dir: Path) -> Path:
    """Download and unpack the small English model when it is not present."""
    target = model_dir / MODEL_NAME
    if (target / "am" / "final.mdl").exists():
        return target
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir / f"{MODEL_NAME}.zip"
    urllib.request.urlretrieve(MODEL_URL, archive)
    try:
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(model_dir)
    finally:
        archive.unlink(missing_ok=True)
    if not (target / "am" / "final.mdl").exists():
        raise RuntimeError("The downloaded Vosk model archive was incomplete.")
    return target


def convert_to_16khz_mono_wav(audio_bytes: bytes) -> bytes:
    """Decode an audio upload and encode mono 16 kHz PCM WAV for Vosk."""
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, int(sample_rate) // divisor)
    converted = io.BytesIO()
    sf.write(converted, mono, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return converted.getvalue()


def transcribe_wav(audio_bytes: bytes, model: Model | None) -> tuple[str | None, str | None]:
    """Transcribe an audio upload locally and return (text, friendly_error)."""
    if not audio_bytes:
        return None, "No audio was received. Please record a short question and try again."
    try:
        normalized_wav = convert_to_16khz_mono_wav(audio_bytes)
        with wave.open(io.BytesIO(normalized_wav), "rb") as source:
            frames = source.readframes(source.getnframes())
        samples = np.frombuffer(frames, dtype="<i2")
        if samples.size == 0 or float(np.sqrt(np.mean(np.square(samples.astype(np.float32))))) < 35:
            return None, "That recording sounds empty or too quiet. Please speak clearly and try again."
        if model is None:
            return None, "Offline speech recognition is not ready. Please retry after the model has loaded."
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        recognizer.AcceptWaveform(frames)
        recognized = json.loads(recognizer.FinalResult()).get("text", "").strip()
        if not recognized:
            return None, "I couldn't make out any speech. Please speak clearly and try again."
        return recognized, None
    except (RuntimeError, ValueError, OSError, sf.SoundFileRuntimeError) as error:
        if isinstance(error, RuntimeError) and "Vosk" in str(error):
            return None, "Offline speech recognition is not ready. Please retry after the model has loaded."
        if isinstance(error, ValueError) and not audio_bytes:
            return None, "No audio was received. Please record a short question and try again."
        return None, "I couldn't process that recording. Please try recording again or type your question."
