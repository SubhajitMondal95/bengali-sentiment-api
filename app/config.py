
import json
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Path to deployment configuration
CONFIG_PATH = BASE_DIR / "deployment_config.json"


# Load deployment configuration
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    deployment_config = json.load(file)


# Model configuration
MODEL_NAME = deployment_config["model_name"]

BASE_MODEL = deployment_config["base_model"]

PROBLEM_TYPE = deployment_config["problem_type"]

MAX_LENGTH = deployment_config["max_length"]


# Label configuration
LABEL_NAMES = deployment_config["labels"]

NUM_LABELS = len(LABEL_NAMES)


# Optimized classification thresholds
THRESHOLDS = [
    deployment_config["thresholds"][label]
    for label in LABEL_NAMES
]


# Basic configuration validation
assert len(THRESHOLDS) == NUM_LABELS

assert all(0 <= threshold <= 1 for threshold in THRESHOLDS)


print("Deployment configuration loaded successfully.")
print("Model:", MODEL_NAME)
print("Labels:", LABEL_NAMES)
print("Thresholds:", THRESHOLDS)
