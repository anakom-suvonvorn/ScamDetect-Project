import pandas as pd
import os
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---

# CHARACTER SETTINGS (Adjust these based on your needs)
# 1 Thai character is roughly equal to 1 char here.
# For BERT models: 512 tokens is approx 1500-2000 Thai characters.
MAX_CHARS = 500     # Maximum size of a chunk (Upper limit of random)
MIN_CHARS = 200        # Minimum size of a chunk (Lower limit of random)
TARGET_OVERLAP = 0.20  # We aim for ~20% overlap between chunks

def smart_split_text(text):
    """
    Splits a long string into random overlapping chunks based on characters.
    Ensures no chunk is smaller than MIN_CHARS.
    """
    text_len = len(text)
    
    # Case 1: Text is short enough to fit in one chunk
    if text_len <= MAX_CHARS:
        return [text]
    
    chunks = []
    start = 0
    
    while True:
        # 1. Pick a random length for this specific chunk
        current_window_size = random.randint(MIN_CHARS, MAX_CHARS)
        
        # 2. Calculate End Point
        end = start + current_window_size
        
        # --- THE "SMART TAIL" LOGIC ---
        # If this chunk goes past the end (or leaves a tiny useless tail),
        # we snap to the very end of the text.
        if end >= text_len:
            # Create a chunk from the very end backwards
            # ensuring it meets the size requirement
            final_chunk = text[-current_window_size:] 
            
            # Prevent duplicates: if this final chunk is identical to the last one we added, skip it.
            if not chunks or final_chunk != chunks[-1]:
                chunks.append(final_chunk)
            break
        
        # 3. Normal Chunking
        chunk = text[start : end]
        chunks.append(chunk)
        
        # 4. Calculate Step Size (Randomized)
        # We want to move forward, but keep some overlap.
        # step = window - overlap
        overlap_amount = int(current_window_size * TARGET_OVERLAP)
        base_step = current_window_size - overlap_amount
        
        # Add "Jitter" to the step so splits aren't predictable
        step_jitter = random.randint(-50, 50) 
        actual_step = max(1, base_step + step_jitter) # Ensure we move forward at least 1 char
        
        start += actual_step
        
        # Safety break if start somehow exceeds length (should be caught by the tail logic, but safe to have)
        if start >= text_len:
            break
            
    return chunks

def process_dataset(input_files, output_files):
    for input_file, output_file in zip(input_files, output_files):
        if not os.path.exists(input_file):
            print(f"❌ Error: {input_file} not found.")
            continue

        data = pd.read_csv(input_file)

        train, test = train_test_split(data, test_size=0.2, random_state=42)

        for idx, df in enumerate([train, test]):
            new_rows = []
        
            print(f"Processing {len(df)} rows...")
            
            for index, row in tqdm(df.iterrows(), total=len(df)):
                original_text = str(row['text'])
                try:
                    label = int(row['label'])
                except:
                    continue
                
                # Handle empty or nan text
                if not original_text or original_text.lower() == 'nan':
                    continue
                    
                # Apply the smart splitter
                splitted_texts = smart_split_text(original_text)
                
                for fragment in splitted_texts:
                    new_rows.append({
                        "text": fragment,
                        "label": label
                    })

            # Save
            final_df = pd.DataFrame(new_rows)
            print(f"✅ Done! Expanded {len(df)} rows into {len(final_df)} training samples.")
            final_df.to_csv(f"{output_file}_{'train' if idx == 0 else 'test'}.csv", index=False)

if __name__ == "__main__":
    input_files = ["original_data/additional_data_2.csv"]
    output_files = ["processing_data/additional_data_2_splitted"]
    process_dataset(input_files, output_files)