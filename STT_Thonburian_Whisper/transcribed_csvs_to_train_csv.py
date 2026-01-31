import pandas as pd
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---

# --- SCAM DATASETS (Comment/Uncomment as needed) ---
# INPUT_FOLDER = "ScamTranscribedLargeFiles/"
# OUTPUT_BASE = "ScamTranscribedLargeFiles/large_transcribed_no_filter_final" # No .csv extension here

# INPUT_FOLDER = "ScamTranscribedMediumFiles/"
# OUTPUT_BASE = "ScamTranscribedMediumFiles/medium_transcribed_no_filter_final"

# INPUT_FOLDER = "additional_data_2/normal_phone_transcribed"
# OUTPUT_BASE = "additional_data_2/normal_phone_data"

INPUT_FOLDER = "additional_data_2/scam_phone_transcribed"
OUTPUT_BASE = "additional_data_2/scam_phone_data"

# --- NORMAL DATASET ---
# INPUT_FOLDER = "audiofiles - call center/right_call-center_trimmed_and_noised_transcribed/"
# OUTPUT_BASE = "audiofiles - call center/right_call-center_trimmed_and_noised_transcribed_final"

# --- SETTINGS ---

# normal
# MIN_WINDOW = 5   # Minimum lines (e.g., short exchange)
# MAX_WINDOW = 12  # Maximum lines (e.g., long explanation)

MIN_WINDOW = 3   # Minimum lines (e.g., short exchange)
MAX_WINDOW = 6  # Maximum lines (e.g., long explanation)

# OVERLAP_RATIO = 0.15 # We aim to overlap roughly 15% of the previous window
OVERLAP_RATIO = 0.3
TEST_SIZE = 0.2  # 20% of files go to the Test Set

# --- FILTER LIST (Optional) ---
# num_list = [24, 28, 56, 39, 69, 15, 51, 40, 29, 71, 61, 1, 67, 50, 5, 73, 64, 47, 59, 75, 19, 7, 41, 22, 74, 25, 13, 20] + [9, 27]
# num_list = [52, 38, 10, 16, 72, 66, 34, 2, 57, 12, 31, 4, 32, 3]
# num_list = [23, 8, 21, 37, 26, 68, 55, 60, 17, 49, 18, 70, 30, 35, 46]


def process_batch(file_list, desc_text="Processing"):
    """
    Inner function to handle the sliding window logic for a specific list of files.
    """
    batch_data = []
    
    for filename in tqdm(file_list, desc=desc_text):
        file_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            df = pd.read_csv(file_path)
            texts = df['text'].astype(str).tolist()
            num_sentences = len(texts)
            
            i = 0
            while i < num_sentences:
                # 1. Random Window Size
                current_window_size = random.randint(MIN_WINDOW, MAX_WINDOW)
                
                # 2. Slice
                window = texts[i : i + current_window_size]
                
                # Skip if remaining chunk is too small (< 2 lines)
                if len(window) < 2 and i > 0:
                    break 

                combined_text = " ".join(window)
                
                # 3. Add Data
                batch_data.append({
                    "text": combined_text, 
                    # "label": 0, # Change to 1 for SCAM datasets
                    "label": 1, # Change to 1 for SCAM datasets
                })
                
                # 4. Calculate Step with Jitter
                base_step = int(current_window_size * (1 - OVERLAP_RATIO))
                step_jitter = random.randint(-1, 1)
                actual_step = max(1, base_step + step_jitter)
                
                i += actual_step

        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            
    return pd.DataFrame(batch_data)

def create_variable_window_dataset():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder '{INPUT_FOLDER}' does not exist.")
        return

    # 1. Gather all files first
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    
    # 2. Apply Filters (If you uncomment your logic)
    valid_files = []
    for f in all_files:
        # Example: Uncomment logic below to filter
        # if f not in [f"scam{num:02d}.csv" for num in num_list]:
        # if f not in [f"scam{num:02d}_conversation.csv" for num in num_list]:
        #     continue
        valid_files.append(f)
        
    print(f"📂 Found {len(valid_files)} files to process.")
    print(valid_files)

    random.shuffle(valid_files)

    # 3. Split TRAIN vs TEST (File-level split to prevent leakage)
    train_files, test_files = train_test_split(valid_files, test_size=TEST_SIZE, random_state=42)
    
    print(f"   - Training on {len(train_files)} file: {train_files}")
    print(f"   - Testing on {len(test_files)} files: {test_files}")

    # 4. Process Both Sets
    train_df = process_batch(train_files, desc_text="Generating Train Data")
    test_df = process_batch(test_files, desc_text="Generating Test Data")
    
    # 5. Save
    # Shuffle rows for better training
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # Output filenames
    # Removes ".csv" if you accidentally put it in OUTPUT_BASE, then adds suffix
    clean_base = OUTPUT_BASE.replace(".csv", "") 
    
    train_out = f"{clean_base}_train.csv"
    test_out = f"{clean_base}_test.csv"
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    
    print(f"✅ Done!")
    print(f"   Saved Train: {train_out} ({len(train_df)} rows)")
    print(f"   Saved Test:  {test_out} ({len(test_df)} rows)")

if __name__ == "__main__":
    create_variable_window_dataset()