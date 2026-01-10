import subprocess
import os
from df.enhance import enhance, init_df, load_audio, save_audio
from df.io import resample

def clean_audio_deepfilter(input_path, output_path): # doesn't work, too bad
    # Load model (downloads automatically on first run)
    model, df_state, _ = init_df()

    # Load and resample to 48000 Hz (Required for DeepFilterNet)
    audio, _ = load_audio(input_path, sr=df_state.sr())

    # Denoise
    enhanced_audio = enhance(model, df_state, audio)

    # Save (it will save at 48k)
    save_audio(output_path, enhanced_audio, df_state.sr())
    
    print(f"Cleaned audio saved to: {output_path}")

def process_call_audio(input_path, final_output_path): # slightly better, still too bad tho
    # Temp file for the normalized version
    normalized_temp = "temp_normalized.wav"

    print("1️⃣ Normalizing Volume (Boosting quiet speaker)...")
    # We use FFmpeg's 'speechnorm' filter which is specifically designed for this.
    # If speechnorm fails (older ffmpeg), use 'dynaudnorm' instead.
    command = [
        "ffmpeg",
        "-y",                     # Overwrite without asking
        "-i", input_path,         # Input
        "-af", "speechnorm=e=12.5:r=0.0001:l=1", # The Magic Filter
        "-ar", "48000",           # Resample to 48k for DeepFilterNet
        normalized_temp
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # Fallback to dynaudnorm if speechnorm is missing
        print("⚠️ 'speechnorm' filter not found, falling back to 'dynaudnorm'")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-af", "dynaudnorm=f=200:g=15", # Aggressive dynamic gain
            "-ar", "48000",
            normalized_temp
        ], check=True)

    print("2️⃣ Removing Noise (DeepFilterNet)...")
    # Now run DeepFilterNet on the LOUD version
    model, df_state, _ = init_df()
    audio, _ = load_audio(normalized_temp, sr=df_state.sr())
    enhanced_audio = enhance(model, df_state, audio)
    
    # Save Final
    save_audio(final_output_path, enhanced_audio, df_state.sr())
    
    # Cleanup temp file
    if os.path.exists(normalized_temp):
        os.remove(normalized_temp)
    
    print(f"✅ Done! Saved to: {final_output_path}")

def clean_with_demucs(input_path, output_folder): # the best one for now
    # Demucs separates the file into "vocals", "drums", "bass", "other".
    # We use the "denoiser" version (htdemucs) which separates "speech" and "noise".
    subprocess.run([
        "demucs",
        "--two-stems", "vocals", # Keep only vocals, discard rest
        "-n", "htdemucs",        # Use the high-quality model
        input_path,
        "-o", output_folder
    ])
    # The output will be in <output_folder>/htdemucs/<filename>/vocals.wav

# clean_audio_deepfilter("ScamVoiceFiles/scam28.wav", "ScamVoiceFiles/scam28_filtered.wav")
# process_call_audio("ScamVoiceFiles/scam28.wav", "ScamVoiceFiles/scam28_filtered.wav")
# clean_with_demucs("ScamVoiceFiles/scam28.wav", "ScamVoiceFiles/")

for i in range(1,77):
 if i in [48,53]:
    continue
 clean_with_demucs(f"ScamVoiceFiles/scam{i:02d}.wav", "ScamVoiceFiles/")