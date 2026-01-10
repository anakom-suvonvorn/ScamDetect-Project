import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_ROOT = "ScamVoiceFiles/htdemucs"           # Where your folders are
DESTINATION = "ScamVoiceFiles/real_scam_conversation"    # Where you want them to go
# DESTINATION = "ScamVoiceFiles/real_scam_background_noise"    # Where you want them to go
os.makedirs(DESTINATION, exist_ok=True)

def gather_noise_files():
    # 1. Count files first just for the progress bar (optional but nice)
    files_to_move = []
    print("Scanning folders...")
    for root, dirs, files in os.walk(SOURCE_ROOT):
        if "vocals.wav" in files:
        # if "no_vocals.wav" in files:
            files_to_move.append(os.path.join(root, "vocals.wav"))

    print(f"Found {len(files_to_move)} conversation files. Copying now...")
    # print(f"Found {len(files_to_move)} noise files. Copying now...")

    # 2. Copy and Rename
    for old_path in tqdm(files_to_move):
        # old_path looks like: htdemucs/call_01/no_vocals.wav
        
        # Get the folder name (e.g., "call_01") to use as ID
        parent_folder = os.path.basename(os.path.dirname(old_path))
        
        # Create new unique name: "call_01_noise.wav"
        new_filename = f"{parent_folder}_conversation.wav"
        # new_filename = f"{parent_folder}_noise.wav"
        new_path = os.path.join(DESTINATION, new_filename)
        
        # Copy the file
        shutil.copy2(old_path, new_path)

    print(f"✅ Done! All noise files are now in '{DESTINATION}'")

if __name__ == "__main__":
    gather_noise_files()