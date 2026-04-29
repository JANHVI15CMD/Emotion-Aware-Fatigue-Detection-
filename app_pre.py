import streamlit as st
import numpy as np
import librosa
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# ─────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Fatigue Detection",
    page_icon="🎤",
    layout="centered"
)

# ─────────────────────────────────────────────────────
# Emotion → Fatigue Score Mapping
# (RAVDESS 8 emotions → fatigue level 0.0–1.0)
# ─────────────────────────────────────────────────────
EMOTION_FATIGUE_MAP = {
    "angry":     0.82,   # High stress = high fatigue
    "fearful":   0.76,
    "sad":       0.72,
    "disgust":   0.65,
    "surprised": 0.42,
    "neutral":   0.28,
    "calm":      0.15,
    "happy":     0.10,
}

EMOTION_EMOJI = {
    "angry":     "😠",
    "fearful":   "😨",
    "sad":       "😢",
    "disgust":   "🤢",
    "surprised": "😲",
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😊",
}

# ─────────────────────────────────────────────────────
# Fuzzy Logic (from model/fuzzy_logic.py)
# ─────────────────────────────────────────────────────
def get_fatigue_level(score):
    if score < 0.3:
        return "Fresh 😊", "success"
    elif score < 0.5:
        return "Slightly Tired 😐", "info"
    elif score < 0.7:
        return "Moderately Fatigued 😓", "warning"
    else:
        return "Exhausted 😴", "error"

# ─────────────────────────────────────────────────────
# Load Pretrained Model (cached — downloads only once)
# Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
# Trained on: RAVDESS dataset (same as your data!)
# Accuracy: 82.23%
# ─────────────────────────────────────────────────────
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

@st.cache_resource(show_spinner="⏳ Loading pretrained model (first time only)...")
def load_model():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return feature_extractor, model

feature_extractor, model = load_model()

# ─────────────────────────────────────────────────────
# Predict Function
# NOTE: This model requires 16kHz audio
# ─────────────────────────────────────────────────────
def predict_fatigue(audio_path):
    # Load at 16kHz (required by wav2vec2)
    speech, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Prepare input
    inputs = feature_extractor(
        speech,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Inference (no gradients needed)
    with torch.no_grad():
        logits = model(**inputs).logits

    # Probabilities for all 8 emotions
    probs = torch.softmax(logits, dim=-1)[0]
    prob_list = probs.tolist()

    # Get emotion labels from model config
    id2label = model.config.id2label

    # Build emotion → probability dict
    emotion_probs = {
        id2label[i].lower(): prob_list[i]
        for i in range(len(prob_list))
    }

    # Top predicted emotion
    top_emotion = max(emotion_probs, key=emotion_probs.get)
    top_confidence = emotion_probs[top_emotion]

    # Weighted fatigue score across all emotions
    fatigue_score = sum(
        emotion_probs.get(emo, 0.0) * score
        for emo, score in EMOTION_FATIGUE_MAP.items()
    )

    return top_emotion, top_confidence, fatigue_score, emotion_probs

# ─────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────
st.title("🎤 Emotion-Aware Voice Fatigue Detection")
st.caption("Powered by Wav2Vec2 (82% accuracy) • Trained on RAVDESS • Fuzzy Logic Output")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    duration = st.slider("Recording Duration (seconds)", 2, 10, 4)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    record_btn = st.button("🎙️ Start Recording", use_container_width=True)

st.markdown("**Or upload a WAV file:**")
uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3"])

audio_path = None

# ─── Record ───
if record_btn:
    fs = 16000   # 16kHz for wav2vec2
    with st.spinner(f"🎤 Recording for {duration}s... बोलो!"):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
    wav.write("temp.wav", fs, recording)
    st.success("✅ Recording complete!")
    audio_path = "temp.wav"

# ─── Upload ───
if uploaded_file is not None:
    with open("temp_upload.wav", "wb") as f:
        f.write(uploaded_file.read())
    audio_path = "temp_upload.wav"
    st.success("✅ File uploaded!")

# ─────────────────────────────────────────────────────
# Prediction & Results
# ─────────────────────────────────────────────────────
if audio_path:
    st.markdown("---")
    st.subheader("🔍 Analysis Result")

    try:
        with st.spinner("Analyzing voice..."):
            top_emotion, confidence, fatigue_score, emotion_probs = predict_fatigue(audio_path)

        fatigue_label, alert_type = get_fatigue_level(fatigue_score)
        emoji = EMOTION_EMOJI.get(top_emotion, "🎙️")

        # ── Top 3 metrics ──
        c1, c2, c3 = st.columns(3)
        c1.metric("Detected Emotion", f"{emoji} {top_emotion.capitalize()}")
        c2.metric("Confidence", f"{confidence * 100:.1f}%")
        c3.metric("Fatigue Score", f"{fatigue_score:.4f}")

        # ── Fatigue progress bar ──
        st.markdown("**Fatigue Level:**")
        st.progress(float(fatigue_score))

        # ── Fatigue category ──
        if alert_type == "success":
            st.success(f"**{fatigue_label}**")
        elif alert_type == "info":
            st.info(f"**{fatigue_label}**")
        elif alert_type == "warning":
            st.warning(f"**{fatigue_label}**")
        else:
            st.error(f"**{fatigue_label}**")

        # ── All emotion probabilities ──
        with st.expander("📊 All Emotion Probabilities"):
            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            for emo, prob in sorted_emotions:
                emo_emoji = EMOTION_EMOJI.get(emo, "🎙️")
                bar_val = int(prob * 100)
                st.markdown(
                    f"**{emo_emoji} {emo.capitalize()}** — {prob*100:.1f}%"
                )
                st.progress(float(prob))

        # ── Fuzzy logic table ──
        with st.expander("🧠 Fuzzy Logic Score Ranges"):
            st.markdown("""
            | Score Range | Fatigue Category |
            |---|---|
            | 0.00 – 0.30 | Fresh 😊 |
            | 0.30 – 0.50 | Slightly Tired 😐 |
            | 0.50 – 0.70 | Moderately Fatigued 😓 |
            | 0.70 – 1.00 | Exhausted 😴 |
            """)
            st.markdown(f"**Your Score: `{fatigue_score:.4f}`** → **{fatigue_label}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.code(str(e))

st.markdown("---")
st.caption("Emotion-Aware Voice Fatigue Detection | MCA Data Science | University of Allahabad")