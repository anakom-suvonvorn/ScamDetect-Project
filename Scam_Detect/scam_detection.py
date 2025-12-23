print("loading dependencies")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import argparse
import shap
import json

def parse_csv_transcription(input_path: str):
    df_transcription = pd.read_csv(input_path)
    text = df_transcription['text'].str.cat(sep='')
    return text

def predict_scam(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    predicted_class_id = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][predicted_class_id].item()

    labels = ["Normal", "Scam"] 
    prediction = labels[predicted_class_id]

    return prediction, confidence

def word_risk_importance(text, model, tokenizer):
    def word_risk_predictor(text):
        if isinstance(text, str):
            text = [text]
        inputs = tokenizer(list(text), return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()
        return probs

    explainer = shap.Explainer(word_risk_predictor, tokenizer)
    shap_values = explainer([text])

    words_risk = shap_values[0, :, 1].data
    scores_risk = shap_values[0, :, 1].values

    df_risk = pd.DataFrame({"Word": words_risk, "Risk_Score": scores_risk})
    top_risky_words = df_risk.sort_values(by="Risk_Score", ascending=False) #.head(15)

    return top_risky_words

def save_results(text, result, score, word_risk, output_path):
    print(f"Message: {text}")
    print(f"Prediction: {result}")
    print(f"Confidence: {score:.2f}")
    print(word_risk.head(15))

    word_risk.columns = ['word', 'risk_score']
    word_risk_serializable = word_risk.to_dict(orient='records')

    output = {
        "text": text,
        "result": result,
        "score": score,
        "word_risk": word_risk_serializable
    }

    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(output, fp, ensure_ascii=False, indent=4)

    print(f"results saved at {output_path}")

def main(args):
    print("started running")

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"using: {device}")

    text_msg = parse_csv_transcription(args.input_file)
    result, score = predict_scam(text_msg, model, tokenizer, device)

    print("analyzing word risk importance:")
    word_risk = word_risk_importance(text_msg, model, tokenizer)

    save_results(text_msg, result, score, word_risk, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scam Detection Model Pipeline")
    parser.add_argument("--input_file", required=True, help="Input the transcripted CSV path")
    parser.add_argument("--output_file", default='./test_ouput/output.json', help="Output file path")
    parser.add_argument("--model_path", default='./trained_scam_model', help="Path to the scam detection model")

    args = parser.parse_args()
    main(args)