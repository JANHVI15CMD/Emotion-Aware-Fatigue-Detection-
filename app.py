import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import scipy.io.wavfile as wav
import sys
import os

st.set_page_config(
    page_title="Voice Fatigue Detection",
    page_icon="🎤",
    layout="centered"
)


def get_fatigue_level(score):
    if score < 0.5:
        return "Fresh ", "success"
    elif score < 0.7:
        return "Slightly Tired ", "info"
    elif score < 0.8:
        return "Moderately Fatigued ", "warning"
    else:
        return "Exhausted ", "error"


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # Same as create_spectrograms.py
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to 128x128
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
    mel_db = mel_db[:128, :]

    #  FIX: Same normalization as training (min-max per file)
    min_val = np.min(mel_db)
    max_val = np.max(mel_db)
    if (max_val - min_val) == 0:
        mel_db = np.zeros_like(mel_db)
    else:
        mel_db = (mel_db - min_val) / (max_val - min_val)

    # Shape: (1, 128, 128, 1)
    features = mel_db.reshape(1, 128, 128, 1).astype("float32")
    return features


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model_best.keras")

model = load_model()


st.title(" Emotion-Aware Voice Fatigue Detection")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    duration = st.slider("Recording Duration (seconds)", 2, 10, 3)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    record_btn = st.button(" Start Recording", use_container_width=True)


st.markdown("**Or upload an audio file:**")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

audio_path = None


if record_btn:
    fs = 22050
    with st.spinner(f"Recording for {duration} seconds..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
    wav.write("temp.wav", fs, recording)
    st.success(" Recording complete!")
    audio_path = "temp.wav"


if uploaded_file is not None:
    with open("temp_upload.wav", "wb") as f:
        f.write(uploaded_file.read())
    audio_path = "temp_upload.wav"
    st.success(" File uploaded!")


if audio_path:
    st.markdown("---")
    st.subheader(" Analysis Result")

    try:
        features = extract_features(audio_path)
        prediction = model.predict(features, verbose=0)
        score = float(prediction[0][0])

        fatigue_label, alert_type = get_fatigue_level(score)

        # Score bar
        st.metric(label="Fatigue Score", value=f"{score:.4f}", delta=None)
        st.progress(score)

        # Result
        if alert_type == "success":
            st.success(f"**{fatigue_label}**")
        elif alert_type == "info":
            st.info(f"**{fatigue_label}**")
        elif alert_type == "warning":
            st.warning(f"**{fatigue_label}**")
        else:
            st.error(f"**{fatigue_label}**")

        # Score breakdown
        with st.expander(" Score Breakdown"):
            st.markdown(f"""
            | Range | Category |
            |---|---|
            | 0.00 – 0.50 | Fresh  |
            | 0.50 – 0.70 | Slightly Tired  |
            | 0.70 – 0.80 | Moderately Fatigued  |
            | 0.80 – 1.00 | Exhausted  |

            **Your score: `{score:.4f}`** → **{fatigue_label}**
            """)

    except Exception as e:
        st.error(f" Prediction Error: {e}")
        st.info("Make sure `cnn_model_best.keras` is in the same folder as app.py")

st.markdown("---")
st.caption("Emotion-Aware Voice Fatigue Detection | MCA Data Science | University of Allahabad")