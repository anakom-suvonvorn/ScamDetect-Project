# Setup

do the `Setup` steps in the `README` file inside both the `Scam_Detect` folder and the `STT_Thonburian_Whisper` folder

also make sure that `uv init .` is ran inside of their own respective folder

# How to use and command examples

using the default models

```
python main_runner.py --input_file test.mp3
```

*when running the command, make sure that the `audio file` is inside the `input` folder

using other models (Speech to Text/Scam Detect model)
```
python main_runner.py --input_file test.mp3 --STT_model_path <stt_model_path> --Scam_Detect_model_path <scam_detection_path>
```
*available thonburian whisper models (hugging face)
- `biodatlab/whisper-th-small-combined`
- `biodatlab/whisper-th-medium-combined` (default)
- `biodatlab/whisper-th-large-combined`
- `biodatlab/whisper-th-large-v3-combined`
- `biodatlab/distill-whisper-th-small`
- `biodatlab/distill-whisper-th-medium`
- `biodatlab/distill-whisper-th-large-v3`

**the scam detect model path is relative to `scam_detection.py` inside of the Scam_Detect folder