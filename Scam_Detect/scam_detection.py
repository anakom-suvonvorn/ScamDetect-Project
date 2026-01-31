print("loading dependencies")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import argparse
import shap
import json
import numpy as np
from collections import defaultdict

def parse_csv_transcription(input_path: str):
    df_transcription = pd.read_csv(input_path)
    text = df_transcription['text'].str.cat(sep='')
    return text

def predict_scam(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=416)
    
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

def calculate_weighted_score(result_score_list, decay_factor=0.6):
    scores = []
    for item in result_score_list:
        if item["result"] == "Scam":
            scores.append(item["score"])
        else:
            scores.append(1.0 - item["score"])

    scores.sort(reverse=True)

    total_weighted_score = 0
    total_weights = 0

    for i, score in enumerate(scores):
        weight = decay_factor ** i  
        
        total_weighted_score += score * weight
        total_weights += weight

    if total_weights == 0: return 0.0
    
    return total_weighted_score / total_weights

def get_chunk_shap_values(chunk_text, chunk_start_offset, model, tokenizer):
    def word_risk_predictor(text):
        if isinstance(text, str):
            text = [text]
        inputs = tokenizer(list(text), return_tensors="pt", padding=True, truncation=True, max_length=416).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()
    
    explainer = shap.Explainer(word_risk_predictor, tokenizer)
    shap_values = explainer([chunk_text])

    tokens = shap_values[0].data
    scores = shap_values[0].values[:, 1]

    # tokens = shap_values[0, :, 1].data
    # scores = shap_values[0, :, 1].values

    encoding = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding['offset_mapping']

    mapped_results = []

    limit = min(len(tokens), len(scores), len(offsets))

    for i in range(limit):
        start, end = offsets[i]
        
        # Skip special tokens that have 0 length or weird formatting if necessary
        if start == end: continue 

        global_start = chunk_start_offset + start
        global_end = chunk_start_offset + end
        
        mapped_results.append({
            "word": tokens[i],
            "risk_score": scores[i],
            "global_start": global_start,
            "global_end": global_end
        })
        
    return mapped_results

def aggregate_shap_risks(all_mapped_results):
    token_accumulator = defaultdict(list)
    token_info = {}

    for item in all_mapped_results:
        g_start = item['global_start']

        token_accumulator[g_start].append(item['risk_score'])

        token_info[g_start] = {
            "word": item['word'],
            "global_end": item['global_end']
        }

    final_risk_list = []
    for start_idx, scores in token_accumulator.items():
        avg_score = sum(scores) / len(scores)

        clean_word = token_info[start_idx]['word'].replace(' ', '')

        final_risk_list.append({
            "position": start_idx,
            "word": clean_word, 
            "risk_score": float(avg_score) # Convert to float for JSON serialization
        })

    df = pd.DataFrame(final_risk_list)
    if not df.empty:
        df = df.sort_values(by="risk_score", ascending=False)
        
    return df

def save_results(text, result, score, df_word_risk, output_path):
    print(f"Message: {text}")
    print(f"Prediction: {result}")
    print(f"Confidence: {score:.2f}")
    print(df_word_risk.head(15))

    # top_risks = df_word_risk.head(50).to_dict(orient='records')
    top_risks = df_word_risk.to_dict(orient='records')

    output = {
        "text": text,
        "result": result,
        "score": score,
        "word_risk": top_risks
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

    length = len(text_msg)
    non_overlap = 0.5
    window_size = 800
    # window_size = 1500
    all_chunks_shap_data = []
    result_score_list = []

    if length > window_size:
        start = 0
        end = window_size
        while True:
            chunk_text = text_msg[start:end]

            print(f"Predicting scam for chunk starting at {start}...")
            result, score = predict_scam(chunk_text, model, tokenizer, device)
            result_score_list.append({"result": result, "score": score})
            print(f"score: {score} ({result})")

            print(f"Analyzing SHAP for chunk starting at {start}...")
            chunk_shap_data = get_chunk_shap_values(chunk_text, start, model, tokenizer)
            all_chunks_shap_data.extend(chunk_shap_data)

            if end == length:
                break
            elif end + int(window_size*non_overlap) > length:
                # start += length - end
                start += int(window_size*non_overlap)
                end = length
            else:
                start += int(window_size*non_overlap)
                end += int(window_size*non_overlap)
    else:
        result, score = predict_scam(text_msg, model, tokenizer, device)

        print("analyzing word risk importance:")
        all_chunks_shap_data = get_chunk_shap_values(text_msg, 0, model, tokenizer)

        result_score_list = [{"result": result, "score": score}]

    print("Doing Final Calculations...")

    final_scam_score = calculate_weighted_score(result_score_list, decay_factor=0.6)
    final_result = "Scam" if final_scam_score > 0.5 else "Normal"

    df_global_risks = aggregate_shap_risks(all_chunks_shap_data)

    save_results(text_msg, final_result, final_scam_score, df_global_risks, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scam Detection Model Pipeline")
    parser.add_argument("--input_file", required=True, help="Input the transcripted CSV path")
    parser.add_argument("--output_file", default='./test_ouput/output.json', help="Output file path")
    parser.add_argument("--model_path", default='./all_data_final_no_data_leakage_v3', help="Path to the scam detection model")

    args = parser.parse_args()
    main(args)