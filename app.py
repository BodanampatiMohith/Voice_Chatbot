"""Streamlit interface for an offline voice-to-intent university chatbot."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from vosk import Model

from src.chatbot import load_artifacts, load_responses, predict_top_intents, reply_for
from src.speech_recognition_utils import ensure_vosk_model, transcribe_wav

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Campus Voice Assistant", page_icon="🎙️", layout="centered")

@st.cache_resource(show_spinner="Loading trained intent model...")
def resources():
    model, tokenizer, label_encoder = load_artifacts()
    responses = load_responses()
    if set(label_encoder.classes_) != set(responses):
        raise FileNotFoundError("The saved model is out of date for this dataset. Run `python train.py` to regenerate the trained artifacts.")
    return model, tokenizer, label_encoder, responses


@st.cache_resource(show_spinner="Preparing offline English speech recognition...")
def offline_speech_model() -> Model:
    return Model(str(ensure_vosk_model(ROOT / "model" / "vosk")))


st.title("Campus Voice Assistant")
st.caption("University information, by voice or text")

try:
    model, tokenizer, label_encoder, responses = resources()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


def handle_message(text: str, source: str, threshold: float) -> None:
    intent, confidence, response = reply_for(text, model, tokenizer, label_encoder, responses, threshold)
    candidates = predict_top_intents(text, model, tokenizer, label_encoder)
    st.session_state.history.append({
        "text": text,
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "source": source,
        "top_intents": candidates,
    })


threshold = st.slider("Confidence threshold", min_value=0.20, max_value=0.95, value=0.30, step=0.05)
input_col, result_col = st.columns([1, 1.15], gap="large")

with input_col:
    st.subheader("Ask a question")
    audio = st.audio_input("Record your question", sample_rate=16_000)
    if audio is not None and st.button("Transcribe and ask", type="primary", use_container_width=True):
        try:
            with st.spinner("Transcribing locally..."):
                speech_model = offline_speech_model()
                recognized_text, error = transcribe_wav(audio.getvalue(), speech_model)
            if error:
                st.warning(error)
            else:
                handle_message(recognized_text or "", "Voice", threshold)
        except (OSError, RuntimeError, ValueError) as error:
            st.error(f"Offline speech recognition could not start: {error}. You can still type your question below.")

    st.markdown("**Or type your question**")
    with st.form("text_question", clear_on_submit=True):
        typed_text = st.text_input("Question", placeholder="e.g. When is the library open?", label_visibility="collapsed")
        submitted = st.form_submit_button("Ask chatbot", use_container_width=True)
    if submitted:
        if typed_text.strip():
            handle_message(typed_text, "Text", threshold)
        else:
            st.warning("Please enter a question first.")

with result_col:
    st.subheader("Latest result")
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.markdown("**Recognized speech**" if latest["source"] == "Voice" else "**Recognized speech / submitted text**")
        st.write(latest["text"])
        if latest["intent"]:
            st.success(f"Intent: {latest['intent']}  |  Confidence: {latest['confidence']:.1%}")
        else:
            st.warning(f"Intent: uncertain  |  Confidence: {latest['confidence']:.1%}")
        st.markdown("**Top 3 intents**")
        for candidate, score in latest["top_intents"]:
            st.write(f"{candidate.replace('_', ' ').title()} · {score:.1%}")
        st.markdown("**Chatbot response**")
        st.info(latest["response"])
    else:
        st.info("Your recognized speech and answer will appear here.")

if st.session_state.history:
    st.divider()
    history_heading, clear_col = st.columns([1, 1])
    history_heading.subheader("Conversation history")
    if clear_col.button("Clear chat", use_container_width=True):
        st.session_state.history.clear()
        st.rerun()
    for item in reversed(st.session_state.history):
        st.markdown(f"**{item['source']} · You:** {item['text']}")
        st.markdown(f"**Assistant:** {item['response']}  ")
        st.caption(f"{item['intent'] or 'Uncertain'} · {item['confidence']:.1%}")
