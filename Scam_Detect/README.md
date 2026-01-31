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

# Output formatting

output example (truncated):
```
{
    "text": "ไม่ว่าเขาจะใช้ตัววิธีไหนจากการเสิร์ชอินเทอร์เน็ตหรือจากการเธอใช้ชัยที่วิถีก็อยากให้เขาทำตัวได้ด้วยตัวเองสักหนึ่งข้อแบบง่าย ๆเพราะว่าหลังสูตรมันเพิ่งชาขึ้นมา มันยังไม่ได้มีอะไรผิดเป็นแท็กเทิร์นให้น้องเห็นชัดเจนว่าเขาจะเรียนอะไร นั่นก็คือสิ่งที่อาจารย์ฟากมามี แค่ นั้น แหละเพราะเดี๋ยวต้องไปฝึกอาจารย์เพิ่มเอางานแรกก่อนเอางาน ของ (...)",
    "result": "Scam",
    "score": 0.9905648896299099,
    "word_risk": [
        {
            "position": 4153,
            "word": "ผม",
            "risk_score": 0.03399882674178419
        },
        {
            "position": 1496,
            "word": "เงินรางวัล",
            "risk_score": 0.0313595590667058
        },
        (...)
    ]
}
```

output is a json file that contains these keys and values
- text: the actual full transcription of the audio
- result: the result of the classification, either being "Scam" or "Normal"
- score: the confidence of the prediction of the model
- word_risk: an array containing word_risk values for each word/token of the text, containing
    - position: the position of the starting charactor of the word/token
    - word: the actual word/token itself
    - risk_score: how much does this word/token contribute to the overall prediction of the text being a scam
    - (the words are sorted and arranged from highest risk contributer to least risk contibuter in the array)