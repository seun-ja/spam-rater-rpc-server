import argparse
import sys
import json
import logging
import torch # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stderr
)

label_map = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

# Load model and tokenizer once at module level
MODEL_DIR = "./merged-model-new"
logging.info("Loading tokenizer from %s", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
logging.info("Loading model from %s", MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
# Optimization: move model to GPU if available and set eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Using device: %s", device)
model = model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logging.info("Running model inference")
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        score = probs[0, pred_idx].item()
    logging.info("Inference complete: label=%s, score=%s", label_map[pred_idx], score)
    return {"label": label_map[pred_idx], "score": score}
