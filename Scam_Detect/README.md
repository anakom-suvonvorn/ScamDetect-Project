# Setup

`uv init .`

replace the generated `myproject.toml` file with this

```
[project]
name = "scam-detect"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "accelerate>=1.12.0",
    "datasets>=4.4.2",
    "ipykernel>=7.1.0",
    "lime>=0.2.0.1",
    "matplotlib>=3.10.8",
    "pandas>=2.3.3",
    "protobuf>=6.33.2",
    "pythainlp>=5.2.0",
    "seaborn>=0.13.2",
    "sentencepiece>=0.2.1",
    "shap>=0.49.1",
    "torch",
    "transformers>=4.57.3",
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch" }
```

`uv sync`

to use the commandline environment, do `.venv\Scripts\activate`

# How to use and command example

```
python scam_detection.py --input_file ../output/test_transcripted.csv --output_file ../output/test_results.json --model_path ./trained_scam_model
```