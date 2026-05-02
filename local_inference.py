import sys
import json
import logging
import os

import torch  # type: ignore
import psutil  # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stderr
)

def log_memory(stage: str):
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024**2
        logging.info("[%s] Memory: %.2f MB", stage, mem)
    except Exception:
        logging.warning("Failed to read memory usage")

MODEL_DIR = "./merged-model-new"
LABEL_MAP = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

# Limit threads (important in containers)
torch.set_num_threads(1)

class ModelCache:
    _instance = None

    def __init__(self):
        if ModelCache._instance is not None:
            raise RuntimeError("ModelCache is a singleton!")
        logging.info("Loading tokenizer from %s", MODEL_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        logging.info("Loading model from %s", MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        # Device selection: prefer MPS (Apple Silicon), then CUDA, then CPU
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            logging.info("Initial device: mps (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("Initial device: cuda (GPU)")
        else:
            self.device = torch.device("cpu")
            logging.info("Initial device: cpu")
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            logging.warning("Failed to move model to %s, falling back to CPU: %s", self.device, e)
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = min(512, getattr(self.tokenizer, "model_max_length", 512))
        logging.info("Using max_length=%s", self.max_length)
        ModelCache._instance = self

    @staticmethod
    def get():
        if ModelCache._instance is None:
            ModelCache()
        return ModelCache._instance

class ModelError(Exception):
    pass

def predict(text: str):
    model_cache = ModelCache.get()
    tokenizer = model_cache.tokenizer
    model = model_cache.model
    device = model_cache.device
    max_length = model_cache.max_length
    
    try:
        log_memory("before_tokenize")

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
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