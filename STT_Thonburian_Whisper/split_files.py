import os
import shutil
import random
import math

# ================= CONFIGURATION =================
# 1. Your Source Folder Path (I pasted your path below)
# Use 'r' before the string to handle backslashes correctly on Windows
SOURCE_DIR = r"D:\All files\Me\university\competition stuff\scamprojectthing\ProjectCode\STT_Thonburian_Whisper\additional_data_2\normal_phone_transcribed"

# 2. Your Desired Split Percentages
# You can change this to [20, 80] or [10, 15, 25, 50] etc.
SPLIT_PCT = [20, 80]

# 3. Mode: Set to 'copy' to keep originals, or 'move' to cut and paste.
OPERATION = 'copy' 
# =================================================

def split_files_into_folders(source, distribution):
    # 1. Validation
    if sum(distribution) != 100:
        print(f"WARNING: Your percentages sum to {sum(distribution)}%, not 100%.")
    
    # 2. Get all files (ignoring sub-folders)
    try:
        all_files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    except FileNotFoundError:
        print("Error: Source directory not found.")
        return

    total_files = len(all_files)
    print(f"Found {total_files} files in directory.")

    # 3. Shuffle files to ensure random distribution
    # This prevents 'date grouping' if your files are sorted by date
    random.shuffle(all_files)

    current_index = 0
    
    # 4. Process each split group
    for i, pct in enumerate(distribution):
        folder_name = os.path.join(source, f"split_{i+1}_{pct}percent")
        
        # Create the sub-folder if it doesn't exist
        os.makedirs(folder_name, exist_ok=True)

        # Calculate how many files belong in this group
        # If it's the LAST group, take all remaining files (handles rounding errors)
        if i == len(distribution) - 1:
            count = total_files - current_index
        else:
            count = int(total_files * (pct / 100))

        # Get the slice of files for this group
        batch = all_files[current_index : current_index + count]
        
        print(f"--> Moving {len(batch)} files to '{folder_name}'...")

        # Move (or copy) the files
        for filename in batch:
            src_path = os.path.join(source, filename)
            dst_path = os.path.join(folder_name, filename)
            
            if OPERATION == 'move':
                shutil.move(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        # Update the index for the next iteration
        current_index += count

    print("\nDone! Files have been split.")

if __name__ == "__main__":
    split_files_into_folders(SOURCE_DIR, SPLIT_PCT)