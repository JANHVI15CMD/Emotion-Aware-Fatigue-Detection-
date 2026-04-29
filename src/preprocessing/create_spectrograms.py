import os
import numpy as np
import pandas as pd
import librosa

csv_path = r"data/processed/features_labeled.csv"
df = pd.read_csv(csv_path)

X = []
y = []

print("Total files:", len(df))

for i, row in df.iterrows():
    file_path = row["file"]
    label = row["label"]

    try:
        audio, sr = librosa.load(file_path, sr=22050)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # ✅ FIX: Single consistent resize to 128x128
        mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
        mel_db = mel_db[:128, :]

        # ✅ FIX: Single per-file normalization to [0, 1]
        min_val = np.min(mel_db)
        max_val = np.max(mel_db)
        if (max_val - min_val) == 0:
            mel_db = np.zeros_like(mel_db)
        else:
            mel_db = (mel_db - min_val) / (max_val - min_val)

        mel_db = mel_db.reshape(128, 128, 1)

        X.append(mel_db)
        y.append(label)

        if i % 100 == 0:
            print(f"Processed {i}/{len(df)} files")

    except Exception as e:
        print("Error:", file_path, e)

X = np.array(X, dtype="float32")
y = np.array(y)

print("Final Shape:", X.shape, y.shape)
print("X range: min={:.4f}, max={:.4f}".format(X.min(), X.max()))

os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X.npy", X)
np.save("data/processed/y.npy", y)

print("✅ DONE! X.npy & y.npy saved")