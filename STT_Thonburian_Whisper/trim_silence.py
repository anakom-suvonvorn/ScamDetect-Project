import os
import torch
import torchaudio
from tqdm import tqdm

# 1. Setup Folders
INPUT_FOLDER = "audiofiles - call center/right_call-center"      # Change this to your folder
OUTPUT_FOLDER = "audiofiles - call center/right_call-center_trimmed" # Where saved files go
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 2. Load Silero VAD Model (High precision speech detector)
print("Loading VAD model...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def process_folder():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.wav', '.mp3'))]
    
    for filename in tqdm(files, desc="Processing Audio"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        try:
            # 3. Read Audio
            # Silero expects 16k or 8k sample rate for best results
            wav = read_audio(input_path, sampling_rate=16000)
            
            # 4. Get Timestamps of Speech
            # threshold: 0.5 is standard. Lower it (0.3) if it cuts off faint voices.
            speech_timestamps = get_speech_timestamps(
                wav, 
                model, 
                sampling_rate=16000, 
                threshold=0.4,  # Tuned slightly lower for your faint calls
                min_speech_duration_ms=250, # Ignore blips shorter than 0.25s
                min_silence_duration_ms=500 # Join chunks if silence is < 0.5s
            )
            
            if len(speech_timestamps) > 0:
                # 5. Merge only the speech parts
                merged_wav = collect_chunks(speech_timestamps, wav)
                
                # 6. Save
                save_audio(output_path, merged_wav, sampling_rate=16000)
            else:
                print(f"⚠️ No speech found in {filename} (Skipping)")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    process_folder()