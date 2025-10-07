import sys
import os

# --- File Paths & Names ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src import config as main_config
from train.classifier.config import MODEL_NAME as CLASSIFIER_MODEL_NAME

# Input model path (the classifier we will be calibrating)
CLASSIFIER_MODEL_PATH = os.path.join(main_config.MODELS_DIR, CLASSIFIER_MODEL_NAME)

# Output model path
MODEL_OUTPUT_DIR = main_config.MODELS_DIR
MODEL_NAME = "confidence_calibrator.pkl"