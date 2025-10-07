import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.logger import logger
from train.classifier.train import run_training
from train.classifier.config import MODEL_OUTPUT_DIR, MODEL_NAME

def train_classifier_if_needed():
   
    logger.info("--- Checking for existing Problem Classifier model ---")
    
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    
    if os.path.exists(model_path):
        logger.info(f"Model '{MODEL_NAME}' already exists at {model_path}. Skipping training.")
    else:
        logger.warning(f"Model '{MODEL_NAME}' not found. Starting training process...")
        try:
            run_training()
        except Exception as e:
            logger.error(f"An error occurred during classifier training: {e}")

if __name__ == "__main__":
    train_classifier_if_needed()