import pandas as pd
import os

# 🔹 Input CSV
input_csv = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\processed\features.csv"

# 🔹 Output CSV
output_csv = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\processed\features_labeled.csv"

df = pd.read_csv(input_csv)

def get_label(file_path):
    filename = os.path.basename(file_path).lower()

    # 🔹 RAVDESS
    if "ravdess" in file_path.lower():
        parts = filename.split("-")

        if len(parts) > 2:
            emotion_code = parts[2]

            if emotion_code in ["04", "05", "06"]:   # sad, angry, fear
                return 1   # fatigue
            else:
                return 0   # non-fatigue

    # 🔹 TESS
    if "tess" in file_path.lower():
        if "sad" in filename or "fear" in filename or "angry" in filename:
            return 1
        elif "happy" in filename or "neutral" in filename:
            return 0

    return None

# 🔹 Apply labels
df["label"] = df["file"].apply(get_label)

# 🔹 Remove unlabeled (OpenSLR remove)
df = df[df["label"].notnull()]

# 🔹 Convert to int
df["label"] = df["label"].astype(int)

# 🔹 Save
df.to_csv(output_csv, index=False)

print("✅ DONE! Labeled dataset ready")

# 🔹 Check balance
print("\n📊 Label Distribution:")
print(df["label"].value_counts())
