
import torch

from scipy.special import expit

from app import model as model_loader

from app.config import (
    LABEL_NAMES,
    THRESHOLDS,
    MAX_LENGTH
)


def predict_text(input_text: str):

    # Check whether model is loaded
    if not model_loader.is_model_loaded():
        raise RuntimeError("Model is not loaded yet.")

    # Tokenize input sentence
    inputs = model_loader.tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    # Move inputs to the model's device
    inputs = {
        key: value.to(model_loader.device)
        for key, value in inputs.items()
    }

    # Generate predictions
    with torch.inference_mode():
        outputs = model_loader.model(**inputs)

    # Convert logits into probabilities using sigmoid
    logits = outputs.logits.cpu().numpy()[0]
    probabilities = expit(logits)

    # Apply individual thresholds
    predictions = {}

    probability_scores = {}

    for i, label in enumerate(LABEL_NAMES):

        score = float(probabilities[i])

        threshold = THRESHOLDS[i]

        predictions[label] = score >= threshold

        probability_scores[label] = round(score, 4)

    return {
        "input_text": input_text,
        "predictions": predictions,
        "probabilities": probability_scores
    }
