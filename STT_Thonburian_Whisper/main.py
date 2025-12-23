import argparse
import torch
import pandas as pd
from datasets import Audio, Dataset
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.pipelines.pt_utils import KeyDataset
from sentence_segment import SyllableSegmentation
from utils import convert_mp4_to_wav, perform_vad, generate_srt, burn_srt_to_video
from pydub import AudioSegment
from tqdm import tqdm


def convert_audio_to_wav(audio_file, target_sr):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    output_wav_file = audio_file.rsplit('.', 1)[0] + "_converted.wav"
    audio.export(output_wav_file, format="wav")
    return output_wav_file

def main(args):
    SAMPLING_RATE = 16000

    # Do ASR
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model on {device}...")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )

    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        device=device,
    )
    
    if args.input_file.endswith('.mp4'):
        wav_file = convert_mp4_to_wav(args.input_file)
    elif args.input_file.endswith('.wav'):
        # Check sampling rate and convert if necessary
        audio = AudioSegment.from_wav(args.input_file)
        if audio.frame_rate != SAMPLING_RATE:
            wav_file = convert_audio_to_wav(args.input_file, SAMPLING_RATE)
        else:
            wav_file = args.input_file
    else:  # Assuming other audio formats such as .mp3, etc.
        wav_file = convert_audio_to_wav(args.input_file, SAMPLING_RATE)

    _, chunklist = perform_vad(wav_file, 'temp_directory_for_chunks')
    
    # for faster inference, create dataset
    audio_dataset = Dataset.from_dict({"audio": [c["fname"] for c in chunklist]}).cast_column("audio", Audio())\
    
    print("audio successfully split into chunks and added into dataset")

    prediction_gen = pipe(
        KeyDataset(audio_dataset, "audio"),
        # generate_kwargs={"task": "transcribe", "language": "Thai"},
        # return_timestamps=False,
        batch_size=4,
        ignore_warning=False,
    )

    predictions = []
    for out in tqdm(prediction_gen, total=len(audio_dataset), miniters=1):
        predictions.append(out)

    print("predictions made (pipeline ran successfully)")

    vad_transcriptions = {"start": [], "end": [], "prediction": []}

    for vad_chunk, pred in zip(chunklist, predictions):
        start_in_samples, end_in_samples = vad_chunk["start"], vad_chunk["end"]
        start_in_s = start_in_samples / (SAMPLING_RATE)
        end_in_s = end_in_samples / (SAMPLING_RATE)

        vad_transcriptions["prediction"].append(pred["text"])
        vad_transcriptions["start"].append(start_in_s)
        vad_transcriptions["end"].append(end_in_s)

        print(f"prediction: {pred['text']}")
        print(f"start: {start_in_s} end: {end_in_s}")

    print("creating final data to save")

    ss = SyllableSegmentation()
    uncorrected_segments = ss(vad_transcriptions=vad_transcriptions, segment_duration=4.0)

    print("saving to file")

    if args.output_format == 'csv':
        df = pd.DataFrame(uncorrected_segments)
        df.to_csv(args.output_file, index=False)
    elif args.output_format == 'srt':
        generate_srt(uncorrected_segments, args.output_file)
        if args.burn_srt:
            burn_srt_to_video(args.input_file, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR pipeline with options")
    parser.add_argument("--input_file", required=True, help="Input video or audio file path")
    parser.add_argument("--output_file", required=True, help="Output file path (CSV or SRT based on the format specified)")
    parser.add_argument("--model_path", default='biodatlab/whisper-th-medium-combined', help="Path to the whisper model")
    parser.add_argument("--output_format", choices=['csv', 'srt'], default='csv', help="Output format, either csv or srt")
    parser.add_argument("--burn_srt", action='store_true', help="Option to burn the srt to the input video (only works if output_format is srt)")

    args = parser.parse_args()
    main(args)