
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from app.config import MODEL_NAME, NUM_LABELS


# Global variables
tokenizer = None
model = None

# Select available device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load model and tokenizer
def load_model():

    global tokenizer, model

    print(f"Loading model: {MODEL_NAME}")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    model.to(device)

    # Set evaluation mode
    model.eval()

    print("Tokenizer and model loaded successfully.")


# Check whether model and tokenizer are loaded
def is_model_loaded():

    return model is not None and tokenizer is not None
