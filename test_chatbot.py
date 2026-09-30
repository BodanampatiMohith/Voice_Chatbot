"""Smoke tests for the trained chatbot artifacts. Run after `python train.py`."""
import io
import wave

import numpy as np

from src.chatbot import load_artifacts, load_responses, reply_for
from src.speech_recognition_utils import transcribe_wav

CASES = {
    "Hello": "greeting", "Namaste, can you help": "greeting",
    "Goodbye": "goodbye", "I am signing off now": "goodbye",
    "Thank you very much": "thanks", "You cleared my doubt": "thanks",
    "How do I apply for admission": "admission", "When is the admission counselling": "admission",
    "What courses are available": "courses", "How long is the BCA programme": "courses",
    "How much is the semester fee": "fees", "Can I pay tuition online": "fees",
    "Who teaches computer science": "faculty", "Where can I find faculty profiles": "faculty",
    "Can I borrow a library book": "library", "How do I renew my library book": "library",
    "Does the university have hostel rooms": "hostel", "What is the hostel mess menu": "hostel",
    "Tell me about campus placements": "placements", "Are mock interviews organised": "placements",
    "Where is the university campus": "campus", "Can I get a campus map": "campus",
    "Where can I see my class timetable": "timetable", "What time is my next lecture": "timetable",
    "When are semester exams": "examinations", "How do I download my hall ticket": "examinations",
    "How can I contact the university": "contact", "What is the official college email": "contact",
    "Are scholarships available": "scholarships", "How do I renew my scholarship": "scholarships",
    "Where can I check my attendance": "attendance", "What is the minimum attendance": "attendance",
    "Where can I see the bus routes": "transport", "How do I apply for a bus pass": "transport",
    "What student clubs are available": "student_activities", "When is the college fest": "student_activities",
}


def silent_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(np.zeros(16_000, dtype="<i2").tobytes())
    return buffer.getvalue()


def main():
    model, tokenizer, encoder = load_artifacts()
    responses = load_responses()
    assert len(CASES) == 36, "Add exactly two test utterances per intent."
    failures = []
    for text, expected in CASES.items():
        actual, confidence, response = reply_for(text, model, tokenizer, encoder, responses)
        print(f"{text!r} -> {actual} ({confidence:.1%}): {response}")
        if actual != expected: failures.append(f"{text}: expected {expected}, got {actual}")
    for question in ["What is the weather forecast on Mars?", "How do I cook pasta?"]:
        unknown_intent, _, _ = reply_for(question, model, tokenizer, encoder, responses)
        assert unknown_intent is None, f"Unrelated question should use the fallback response: {question}"
    empty_intent, _, _ = reply_for("   ", model, tokenizer, encoder, responses)
    assert empty_intent is None, "Empty text should not be classified"
    speech, error = transcribe_wav(b"", None)
    assert speech is None and error, "Empty audio should produce a graceful speech-recognition error"
    speech, error = transcribe_wav(silent_wav(), None)
    assert speech is None and error and "quiet" in error, "Silent audio should produce a friendly speech-recognition error"
    assert not failures, "\n".join(failures)
    print("All smoke tests passed.")

if __name__ == "__main__":
    main()
