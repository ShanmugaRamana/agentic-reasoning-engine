import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.logger import logger
# Import for the classifier
from train.classifier.train import run_training as run_classifier_training
from train.classifier.config import MODEL_OUTPUT_DIR as CLASSIFIER_MODEL_DIR, MODEL_NAME as CLASSIFIER_MODEL_NAME

# Import for the analogical reasoner
from train.analogy.train import run_training as run_analogy_training
from train.analogy.config import MODEL_OUTPUT_DIR as ANALOGY_MODEL_DIR, MODEL_NAME as ANALOGY_MODEL_NAME

def train_classifier_if_needed():
    """
    Checks if the problem classifier model exists and trains it if it doesn't.
    """
    logger.info("--- Checking for existing Problem Classifier model ---")
    
    model_path = os.path.join(CLASSIFIER_MODEL_DIR, CLASSIFIER_MODEL_NAME)
    
    if os.path.exists(model_path):
        logger.info(f"✅ Model '{CLASSIFIER_MODEL_NAME}' already exists. Skipping training.")
    else:
        logger.warning(f"⚠️ Model '{CLASSIFIER_MODEL_NAME}' not found. Starting training process...")
        try:
            run_classifier_training()
        except Exception as e:
            logger.error(f"❌ An error occurred during classifier training: {e}")

def train_analogy_reasoner_if_needed():
    """
    Checks if the analogical reasoner model exists and builds it if it doesn't.
    """
    logger.info("--- Checking for existing Analogical Reasoner model ---")
    
    model_path = os.path.join(ANALOGY_MODEL_DIR, ANALOGY_MODEL_NAME)

    if os.path.exists(model_path):
        logger.info(f"✅ Model '{ANALOGY_MODEL_NAME}' already exists. Skipping training.")
    else:
        logger.warning(f"⚠️ Model '{ANALOGY_MODEL_NAME}' not found. Starting training process...")
        try:
            run_analogy_training()
        except Exception as e:
            logger.error(f"❌ An error occurred during analogical reasoner training: {e}")

if __name__ == "__main__":
    train_classifier_if_needed()
    train_analogy_reasoner_if_needed()