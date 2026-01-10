import os
import sys
import subprocess
from pathlib import Path
import random

BASE_DIR = Path(os.getcwd())
# INPUT_DIR = BASE_DIR / "ScamVoiceFiles"
INPUT_DIR = BASE_DIR / "ScamVoiceFiles" / "real_scam_conversation"
# INPUT_DIR = BASE_DIR / "audiofiles - call center" / "right_call-center_trimmed_and_noised"
# OUTPUT_DIR = BASE_DIR / "ScamTranscribedMediumFiles"
# OUTPUT_DIR = BASE_DIR / "ScamTranscribedLargeFiles"
OUTPUT_DIR = BASE_DIR / "ScamVoiceFiles" / "real_scam_conversation_transcribed"
# OUTPUT_DIR = BASE_DIR / "audiofiles - call center" / "right_call-center_trimmed_and_noised_transcribed"
MODEL_PATH = "biodatlab/whisper-th-medium-combined"
# MODEL_PATH = "biodatlab/whisper-th-large-v3-combined"

if not OUTPUT_DIR.exists():
    print(f"Creating output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

wav_files = list(INPUT_DIR.glob("*.wav"))

if not wav_files:
    print(f"No .wav files found in {INPUT_DIR}")
    sys.exit(1)

print(f"Found {len(wav_files)} audio files to process.")

for index, audio_file in enumerate(wav_files, 1):
    # since there's way too much, let's randomize to only transcribe some of the audio
    # if (random.randint(1, 10) != 1):
    #     continue

    # audio_file = Path(*audio_file.parts[-2:])
    audio_file = Path(*audio_file.parts[-3:])

    output_filename = audio_file.stem + ".csv"
    # output_path = Path(*OUTPUT_DIR.parts[-1:]) / output_filename
    output_path = Path(*OUTPUT_DIR.parts[-2:]) / output_filename

    print(f"[{index}/{len(wav_files)}] Processing: {str(audio_file)} into {str(output_path)}...")

    command = [
        "uv", "run",
        # "--directory", "STT_Thonburian_Whisper",
        "python", "main.py",
        "--input_file", str(audio_file).replace("\\", "/"),
        "--output_file", str(output_path).replace("\\", "/"),
        "--model_path", MODEL_PATH
    ]

    try:
        subprocess.run(command, check=True)
        print(f"   -> Saved to: {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"   -> ERROR processing {audio_file.name}: {e}")
    except Exception as e:
        print(f"\n Error: {e}")
        break