import numpy as np
import librosa
import tensorflow as tf
import os

# 🔹 Load trained model

# 🔹 Fuzzy Logic function
def get_fatigue_level(score):

    if score < 0.3:
        return "Fresh 😊"

    elif score < 0.5:
        return "Slightly Tired 😐"

    elif score < 0.7:
        return "Moderately Fatigued 😓"

    else:
        return "Exhausted 😴"


# 🔹 Predict function
def predict_audio(file_path):
    model = tf.keras.models.load_model("cnn_model_best.keras")

    print("Processing:", file_path)

    # 🔹 Load audio
    y, sr = librosa.load(file_path, sr=None)

    # 🔹 Create spectrogram
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # 🔹 Resize to 128x128
    spec_db = spec_db[:128, :128]
    spec_db = np.pad(spec_db, ((0, max(0, 128 - spec_db.shape[0])),
                               (0, max(0, 128 - spec_db.shape[1]))))

    # 🔹 Reshape for CNN
    spec_db = spec_db[np.newaxis, ..., np.newaxis]

    # 🔹 Predict
    score = model.predict(spec_db)[0][0]

    print(f"\n🎯 Fatigue Score: {score:.2f}")
    print("🧠 Result:", get_fatigue_level(score))


# 🔹 TEST (अपना audio path डालो)
if __name__ == "__main__":

    test_audio = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\raw\english\ravdess\Actor_07\03-01-04-02-02-01-07.wav"

    predict_audio(test_audio)
