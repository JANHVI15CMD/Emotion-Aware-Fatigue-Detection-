import os
import glob
import librosa
import numpy as np
import pandas as pd

#  RAW folder
input_folder = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\raw"

#  OUTPUT CSV
output_csv = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\processed\features.csv"

#  Find all WAV files
audio_files = glob.glob(os.path.join(input_folder, "**", "*.wav"), recursive=True)

print(f" Total WAV files found: {len(audio_files)}\n")

data = []

for file in audio_files:
    try:
        print("Processing:", file)

        y, sr = librosa.load(file, sr=None)

            # MFCC (13 values)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Other features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        # Store ALL features
        data.append([file, *mfccs_mean, zcr, rms])


    except Exception as e:
        print(" Error:", file, e)

columns = ["file"] + [f"mfcc_{i}" for i in range(13)] + ["zcr", "rms"]
df = pd.DataFrame(data, columns=columns)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

df.to_csv(output_csv, index=False)

print("\n DONE! features.csv created")
