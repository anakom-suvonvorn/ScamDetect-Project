# Setup
`uv init .`

replace the generated `myproject.toml` file with this

```
[project]
name = "stt-thonburian-whisper"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
    "accelerate>=1.12.0",
    "attacut>=1.0.6",
    "datasets==2.19.1",
    "pandas>=2.3.3",
    "pyarrow==15.0.2",
    "pydub>=0.25.1",
    "soundfile>=0.13.1",
    "ssg>=0.0.8",
    "transformers==4.41.2",
    "torch==2.3.1",
    "torchaudio==2.3.1",
    "librosa>=0.11.0",
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
torchaudio = { index = "pytorch" }
```

`uv sync`

to use the commandline environment, do `.venv\Scripts\activate`

and then make sure `ffmpeg` is installed system wide, if you are intending to burn subtitles onto video files

# How to use and command examples

command transcription normal (to csv)

```
python main.py --input_file testing_material/test.mp3 --output_file testing_results/transcripted.csv --model_path biodatlab/whisper-th-medium-combined 
```

command transcription normal (to srt)

```
python main.py --input_file testing_material/test2.mp3 --output_file testing_results/transcripted2.srt --output_format srt --model_path biodatlab/whisper-th-medium-combined 
```

command transcription burn video with subtitle (to srt)

```
python main.py --input_file videotest.mp4 --output_file transcripted.srt --model_path biodatlab/whisper-th-medium-combined --output_format srt --burn_srt
```

**unrelated:** command manual burn video with subtitle

```
ffmpeg -i "videotest.mp4" -y -vf subtitles="transcripted.srt" "videotest_with_subtitles.mp4"
