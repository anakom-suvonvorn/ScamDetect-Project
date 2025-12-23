import subprocess
import json
import os
import argparse

def run_stt(input_file, model_path):
    input_path = f"../input/{input_file}"
    output_path = f"../output/{input_file.split(".")[0]}_transcripted.csv"

    print("Running STT...")
    subprocess.run(
        [
            "uv", "run",
            "--directory", "STT_Thonburian_Whisper",
            "python", "main.py",
            "--input_file", input_path,
            "--output_file", output_path,
            "--model_path", model_path
        ],
        check=True
    )
    
    return output_path

def run_scam_detect(transcribed_csv_path, original_audio_file, model_path):
    output_path = f"../output/{original_audio_file.split(".")[0]}_results.json"

    print("Running Scam Detection...")
    subprocess.run(
        [
            "uv", "run",
            "--directory", "Scam_Detect",
            "python", "scam_detection.py",
            "--input_file", transcribed_csv_path,
            "--output_file", output_path,
            "--model_path", model_path
        ],
        check=True
    )
    
    return output_path

def main(args):
    transcribed_csv_path = run_stt(args.input_file, args.STT_model_path)
    results_json_path = run_scam_detect(transcribed_csv_path, args.input_file, args.Scam_Detect_model_path)

    print("Pipeline done")
    print(f"Transcription saved at {transcribed_csv_path}")
    print(f"Results saved at {results_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Transcription and Scam Detection Pipeline")
    parser.add_argument("--input_file", required=True, help="Input an audio (mp3,wav) or video (mp4) file name within the input folder")
    parser.add_argument("--STT_model_path", default='biodatlab/whisper-th-medium-combined', help="Path for the Speech To Text model")
    parser.add_argument("--Scam_Detect_model_path", default='./trained_scam_model', help="Path for the Scam Detection model (referenced from inside Scam_Detect folder)")

    args = parser.parse_args()
    main(args)