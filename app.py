"""Streamlit interface for an offline voice-to-intent university chatbot."""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from vosk import Model

from src.chatbot import load_artifacts, load_responses, predict_top_intents, reply_for
from src.speech_recognition_utils import ensure_vosk_model, transcribe_wav

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Campus Voice Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Chat bubbles */
[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 4px; }

/* TTS pulsing badge */
.tts-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(90deg, #1f77b4, #2196f3);
    color: white; padding: 8px 18px; border-radius: 24px;
    font-size: 15px; font-weight: 600;
    animation: pulse 1.4s ease-in-out infinite;
    box-shadow: 0 2px 10px rgba(33,150,243,0.4);
}
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.75; transform:scale(0.97); } }

/* Context chip */
.ctx-chip {
    background:#f0f4ff; color:#3a5ecf; padding:2px 10px;
    border-radius:12px; font-size:12px; border:1px solid #c5d0f5;
}
</style>
""", unsafe_allow_html=True)


# ── TTS helpers ──────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    safe = html.escape(text, quote=True)
    components.html(
        f"""<script>
          var s = window.parent.speechSynthesis;
          s.cancel();
          var u = new window.parent.SpeechSynthesisUtterance("{safe}");
          u.rate = 0.95; u.pitch = 1.0;
          s.speak(u);
        </script>""",
        height=0,
    )


def stop_speaking() -> None:
    components.html(
        "<script>window.parent.speechSynthesis.cancel();</script>",
        height=0,
    )


# ── Resource loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading intent model...")
def resources():
    m, tok, le = load_artifacts()
    resp = load_responses()
    if set(le.classes_) != set(resp):
        raise FileNotFoundError("Model is out of date. Run `python train.py`.")
    return m, tok, le, resp


@st.cache_resource(show_spinner="Loading speech recognition...")
def offline_speech_model() -> Model:
    return Model(str(ensure_vosk_model(ROOT / "model" / "vosk")))


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("history", []), ("speaking", False), ("last_response", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

try:
    model, tokenizer, label_encoder, responses = resources()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    threshold = st.slider(
        "Confidence threshold", 0.20, 0.95, 0.30, 0.05,
        help="Lower = answers more but may be less accurate",
    )
    st.divider()

    tts_enabled = st.toggle("🔊 Speak responses aloud", value=True)
    if tts_enabled and st.session_state.last_response:
        r_col, s_col = st.columns(2)
        if r_col.button("▶ Replay", use_container_width=True, help="Replay last response"):
            speak(st.session_state.last_response)
            st.session_state.speaking = True
        if s_col.button("⏹ Stop", use_container_width=True, help="Stop speaking"):
            stop_speaking()
            st.session_state.speaking = False

    st.divider()
    if st.session_state.history:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            stop_speaking()
            st.session_state.history.clear()
            st.session_state.last_response = ""
            st.session_state.speaking = False
            st.rerun()

    st.divider()
    st.caption("🏛️ Campus Voice Assistant  \nOffline · Privacy-first AI")


# ── Message handler ───────────────────────────────────────────────────────────
def handle_message(text: str, source: str) -> dict:
    intent, confidence, response = reply_for(
        text, model, tokenizer, label_encoder, responses, threshold
    )
    used_context = False
    context_topic: str | None = None

    # Context fallback: if no intent, try augmenting with previous successful topic
    if intent is None:
        for item in reversed(st.session_state.history):
            if item.get("intent"):
                context_topic = item["intent"]
                break
        if context_topic:
            augmented = f"{context_topic.replace('_', ' ')} {text}"
            i2, c2, r2 = reply_for(
                augmented, model, tokenizer, label_encoder, responses, threshold
            )
            if i2:
                intent, confidence, response = i2, c2, r2
                used_context = True

    candidates = predict_top_intents(text, model, tokenizer, label_encoder)
    entry = {
        "text": text,
        "intent": intent,
        "confidence": confidence,
        "response": response,
        "source": source,
        "top_intents": candidates,
        "used_context": used_context,
        "context_topic": context_topic if used_context else None,
    }
    st.session_state.history.append(entry)
    return entry


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎙️ Campus Voice Assistant")
st.caption("Ask about admissions, fees, courses, hostel, placements, library, and more")

# ── TTS speaking indicator ────────────────────────────────────────────────────
if tts_enabled and st.session_state.speaking:
    ind_col, stop_col = st.columns([3, 1])
    ind_col.markdown('<span class="tts-badge">🔊&nbsp; Speaking...</span>', unsafe_allow_html=True)
    if stop_col.button("⏹ Stop", key="stop_top", use_container_width=True, type="secondary"):
        stop_speaking()
        st.session_state.speaking = False
        st.rerun()

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.info("👋 Start a conversation! Use the **Voice** tab to record or **Type** a question below.")
else:
    for item in st.session_state.history:
        icon = "🎤" if item["source"] == "Voice" else "⌨️"
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(f"{icon} **{item['text']}**")

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(item["response"])
            meta_cols = st.columns([3, 2])
            if item["intent"]:
                meta_cols[0].caption(
                    f"🏷️ **{item['intent'].replace('_', ' ').title()}** · {item['confidence']:.0%} confident"
                )
            else:
                meta_cols[0].caption("🏷️ Intent: *uncertain*")
            if item.get("used_context"):
                meta_cols[1].markdown(
                    f'<span class="ctx-chip">↩ topic: {item["context_topic"].replace("_", " ")}</span>',
                    unsafe_allow_html=True,
                )

st.divider()

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask a question")
voice_tab, text_tab = st.tabs(["🎤 Voice", "⌨️ Type"])

with voice_tab:
    audio = st.audio_input("Record your question", sample_rate=16_000)
    if audio is not None:
        if st.button("🎤 Transcribe & Ask", type="primary", use_container_width=True):
            try:
                with st.spinner("Transcribing..."):
                    speech_model = offline_speech_model()
                    recognized_text, err = transcribe_wav(audio.getvalue(), speech_model)
                if err:
                    st.warning(err)
                else:
                    entry = handle_message(recognized_text or "", "Voice")
                    st.session_state.last_response = entry["response"]
                    if tts_enabled:
                        speak(entry["response"])
                        st.session_state.speaking = True
                    st.rerun()
            except (OSError, RuntimeError, ValueError) as e:
                st.error(f"Speech recognition error: {e}")
    else:
        st.caption("Press the microphone above, ask your question, then click **Transcribe & Ask**.")

with text_tab:
    with st.form("text_form", clear_on_submit=True):
        typed = st.text_input(
            "Your question",
            placeholder="e.g. What are the hostel facilities?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("⌨️ Ask", type="primary", use_container_width=True)
    if submitted:
        if typed.strip():
            entry = handle_message(typed, "Text")
            st.session_state.last_response = entry["response"]
            if tts_enabled:
                speak(entry["response"])
                st.session_state.speaking = True
            st.rerun()
        else:
            st.warning("Please type a question first.")
