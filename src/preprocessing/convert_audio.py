import os
import glob
from pydub import AudioSegment

# 🔹 Input folder (mp3 files)
input_folder = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\raw\hindi\openslr"

# 🔹 Output folder (wav files)
output_folder = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\processed\hindi\openslr"

os.makedirs(output_folder, exist_ok=True)

# 🔹 Find all mp3 files
mp3_files = glob.glob(os.path.join(input_folder, "**", "*.mp3"), recursive=True)

print(f"Total MP3 files: {len(mp3_files)}\n")

# 🔹 Convert
for file_path in mp3_files:
    try:
        print("Converting:", file_path)

        audio = AudioSegment.from_mp3(file_path)

        file_name = os.path.basename(file_path).replace(".mp3", ".wav")
        save_path = os.path.join(output_folder, file_name)

        audio.export(save_path, format="wav")

    except Exception as e:
        print("Error:", file_path, e)

print("\n✅ DONE! All files converted to WAV")
