import os
import soundfile as sf
import librosa
from audiomentations import Compose, AddBackgroundNoise, BandPassFilter, Resample
from tqdm import tqdm

# --- CONFIGURATION ---
CLEAN_AUDIO_FOLDER = "audiofiles - call center/right_call-center_trimmed/"       # Your clean, non-scam audio
NOISE_SOURCE_FOLDER = "ScamVoiceFiles/real_scam_background_noise/"      # Folder containing ONLY your no_vocals.wav files
OUTPUT_FOLDER = "audiofiles - call center/right_call-center_trimmed_and_noised/"  # Result
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- THE RECIPE ---
augment = Compose([
    # 1. Overlay your REAL scam noise
    # sounds_path: points to the folder with your no_vocals.wav files
    # min_snr_db: Signal-to-Noise Ratio. 
    #   - 30 dB = Voice is very clear, noise is faint
    #   - 10 dB = Voice is fighting the noise (bad connection)
    AddBackgroundNoise(
        sounds_path=NOISE_SOURCE_FOLDER,
        min_snr_db=10, 
        max_snr_db=25, 
        noise_rms="relative", # adjust noise volume relative to voice volume
        p=1.0  # Apply to every file
    ),

    # 2. Make the CLEAN voice sound like a phone (Bandpass)
    # Essential because your clean audio has too much bass/treble for a phone call
    BandPassFilter(
        min_center_freq=300, 
        max_center_freq=3400, 
        p=1.0
    ),
    
    # 3. Downsample to Phone Quality (8kHz)
    Resample(
        min_sample_rate=8000, 
        max_sample_rate=8000, 
        p=1.0
    )
])

def process_augmentation():
    # Get list of clean files
    files = [f for f in os.listdir(CLEAN_AUDIO_FOLDER) if f.endswith(('.wav', '.mp3'))]
    
    for filename in tqdm(files, desc="Injecting Real Noise"):
        file_path = os.path.join(CLEAN_AUDIO_FOLDER, filename)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        
        try:
            # Load Clean Audio
            audio, sr = librosa.load(file_path, sr=None)
            
            # Apply Augmentation
            # The library automatically:
            # 1. Picks a random file from NOISE_SOURCE_FOLDER
            # 2. Loops it if it's too short, or cuts it if it's too long
            # 3. Mixes it at the SNR level you set
            augmented_audio = augment(samples=audio, sample_rate=sr)
            
            # Save at 8000Hz (Phone standard)
            sf.write(save_path, augmented_audio, 8000)
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    # Check if noise folder actually has files
    if not os.listdir(NOISE_SOURCE_FOLDER):
        print(f"❌ Error: {NOISE_SOURCE_FOLDER} is empty!")
    else:
        process_augmentation()