import sys
import json
import logging
import os

import torch  # type: ignore
import psutil  # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stderr
)

# ---------------------------
# Utilities
# ---------------------------
def log_memory(stage: str):
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024**2
        logging.info("[%s] Memory: %.2f MB", stage, mem)
    except Exception:
        logging.warning("Failed to read memory usage")

# ---------------------------
# Config
# ---------------------------
MODEL_DIR = "./merged-model-new"
LABEL_MAP = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

# Limit threads (important in containers)
torch.set_num_threads(1)

# ---------------------------
# Load tokenizer & model
# ---------------------------
logging.info("Loading tokenizer from %s", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

logging.info("Loading model from %s", MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info("Initial device: %s", device)

# Safe device move
try:
    model = model.to(device)
except Exception as e:
    logging.warning("Failed to move model to CUDA, falling back to CPU: %s", e)
    device = torch.device("cpu")
    model = model.to(device)

model.eval()

# Determine safe max length
MAX_LENGTH = min(512, getattr(tokenizer, "model_max_length", 512))
logging.info("Using max_length=%s", MAX_LENGTH)

# ---------------------------
# Custom error
# ---------------------------
class ModelError(Exception):
    pass

# ---------------------------
# Prediction function
# ---------------------------
def predict(text: str):
    try:
        log_memory("before_tokenize")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        log_memory("after_tokenize")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        log_memory("after_to_device")

        logging.info("Running model inference")

        with torch.inference_mode():
            log_memory("before_forward")

            outputs = model(**inputs)

            log_memory("after_forward")

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).item()
            score = probs[0, pred_idx].item()

        label = LABEL_MAP.get(pred_idx, f"LABEL_{pred_idx}")

        logging.info(
            "Inference complete: label=%s, score=%s",
            label,
            score
        )

        return {
            "label": label,
            "score": score
        }

    except RuntimeError as e:
        msg = str(e).lower()

        if "out of memory" in msg:
            logging.error("OOM during inference: %s", e)

            # Try to recover CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise ModelError(json.dumps({
                "message": "MODEL_OOM",
                "detail": str(e)
            }))
        else:
            logging.exception("Runtime error during inference")
            raise ModelError(json.dumps({
                "message": "MODEL_FATAL",
                "detail": str(e)
            }))

    except Exception as e:
        logging.exception("Unhandled exception during inference")
        raise ModelError(json.dumps({
            "message": "MODEL_FATAL",
            "detail": str(e)
        }))