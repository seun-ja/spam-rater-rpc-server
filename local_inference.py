import argparse
import sys
import json
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stderr
)

label_map = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

def predict(text, model_dir):
    logging.info("Loading tokenizer from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logging.info("Loading model from %s", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    logging.info("Running model inference")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        score = probs[0, pred_idx].item()
    logging.info("Inference complete: label=%s, score=%s", label_map[pred_idx], score)
    return {"label": label_map[pred_idx], "score": score}

def main():
    parser = argparse.ArgumentParser(
        description="Run local inference with merged-model-new."
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Input text to classify."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="merged-model-new",
        help="Path to model directory.",
    )
    args = parser.parse_args()
    try:
        result = predict(args.text, args.model_dir)
        print(json.dumps(result))  # Only the result goes to stdout
    except Exception as e:
        import traceback
        logging.error(e)
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()