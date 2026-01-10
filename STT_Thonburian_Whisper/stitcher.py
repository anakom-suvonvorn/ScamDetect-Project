import os
import glob
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import re

# --- CONFIGURATION ---
INPUT_ROOT = "sound_sentence"          # Your dataset folder
OUTPUT_FOLDER = "stitched_conversations" # Where the long files go
TARGET_SR = 16000                      # Standardize sample rate (safe choice)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def natural_sort_key(s):
    """
    Sorts strings with numbers naturally (e.g., 'file2.wav' comes before 'file10.wav')
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def stitch_files():
    # 1. Get all ID folders (ID01, ID02, etc.)
    # We filter for directories only
    id_folders = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    
    for id_folder in tqdm(id_folders, desc="Stitching Conversations"):
        full_id_path = os.path.join(INPUT_ROOT, id_folder)
        
        # 2. Find ALL .wav files recursively inside this ID folder
        # os.walk goes into every subfolder (like '0/', '1/', etc.) automatically
        wav_files = []
        for root, dirs, files in os.walk(full_id_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        if not wav_files:
            continue # Skip empty folders
            
        # 3. Sort them correctly by number
        # This fixes the "1, 10, 100, 2" sorting problem
        wav_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
        
        # 4. Load and Concatenate
        combined_audio = []
        
        for wav_path in wav_files:
            try:
                # Load with librosa to force consistent sample rate (prevents crashes)
                # sr=TARGET_SR ensures all parts match
                audio, _ = librosa.load(wav_path, sr=TARGET_SR)
                
                # Optional: Add a tiny bit of silence (0.1s) between sentences 
                # to make it sound more natural? (Uncomment if you want this)
                # silence = np.zeros(int(0.1 * TARGET_SR))
                # combined_audio.append(silence)
                
                combined_audio.append(audio)
            except Exception as e:
                print(f"⚠️ Error reading {wav_path}: {e}")

        # 5. Glue them together
        if combined_audio:
            final_wave = np.concatenate(combined_audio)
            
            # 6. Save as one big file
            output_filename = f"{id_folder}.wav"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            sf.write(output_path, final_wave, TARGET_SR)

if __name__ == "__main__":
    stitch_files()