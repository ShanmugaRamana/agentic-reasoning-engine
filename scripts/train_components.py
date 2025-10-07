import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.logger import logger
# Import for the classifier
from train.classifier.train import run_training as run_classifier_training
from train.classifier.config import MODEL_OUTPUT_DIR as CLASSIFIER_DIR, MODEL_NAME as CLASSIFIER_NAME

# Import for the analogical reasoner
from train.analogy.train import run_training as run_analogy_training
from train.analogy.config import MODEL_OUTPUT_DIR as ANALOGY_DIR, MODEL_NAME as ANALOGY_NAME

# Import for the confidence calibrator
from train.calibrator.train import run_training as run_calibrator_training
from train.calibrator.config import MODEL_OUTPUT_DIR as CALIBRATOR_DIR, MODEL_NAME as CALIBRATOR_NAME

# Import for the meta-reasoner
from train.meta_reasoner.train import run_training as run_meta_reasoner_training
from train.meta_reasoner.config import MODEL_OUTPUT_DIR as META_DIR, MODEL_NAME as META_NAME


def train_classifier_if_needed():
    """Checks if the problem classifier model exists and trains it if it doesn't."""
    logger.info("--- Checking for existing Problem Classifier model ---")
    model_path = os.path.join(CLASSIFIER_DIR, CLASSIFIER_NAME)
    if os.path.exists(model_path):
        logger.info(f"✅ Model '{CLASSIFIER_NAME}' already exists. Skipping training.")
    else:
        logger.warning(f"⚠️ Model '{CLASSIFIER_NAME}' not found. Starting training process...")
        try:
            run_classifier_training()
        except Exception as e:
            logger.error(f"❌ An error occurred during classifier training: {e}")

def train_analogy_reasoner_if_needed():
    """Checks if the analogical reasoner model exists and builds it if it doesn't."""
    logger.info("--- Checking for existing Analogical Reasoner model ---")
    model_path = os.path.join(ANALOGY_DIR, ANALOGY_NAME)
    if os.path.exists(model_path):
        logger.info(f"✅ Model '{ANALOGY_NAME}' already exists. Skipping training.")
    else:
        logger.warning(f"⚠️ Model '{ANALOGY_NAME}' not found. Starting training process...")
        try:
            run_analogy_training()
        except Exception as e:
            logger.error(f"❌ An error occurred during analogical reasoner training: {e}")

def train_calibrator_if_needed():
    """Checks if the confidence calibrator model exists and trains it if it doesn't."""
    logger.info("--- Checking for existing Confidence Calibrator model ---")
    model_path = os.path.join(CALIBRATOR_DIR, CALIBRATOR_NAME)
    if os.path.exists(model_path):
        logger.info(f"✅ Model '{CALIBRATOR_NAME}' already exists. Skipping training.")
    else:
        logger.warning(f"⚠️ Model '{CALIBRATOR_NAME}' not found. Starting training process...")
        try:
            run_calibrator_training()
        except Exception as e:
            logger.error(f"❌ An error occurred during confidence calibrator training: {e}")

def train_meta_reasoner_if_needed():
    """Checks if the meta-reasoner model exists and trains it if it doesn't."""
    logger.info("--- Checking for existing Meta-Reasoner model ---")
    model_path = os.path.join(META_DIR, META_NAME)
    if os.path.exists(model_path):
        logger.info(f"✅ Model '{META_NAME}' already exists. Skipping training.")
    else:
        logger.warning(f"⚠️ Model '{META_NAME}' not found. Starting training process...")
        try:
            run_meta_reasoner_training()
        except Exception as e:
            logger.error(f"❌ An error occurred during meta-reasoner training: {e}")


if __name__ == "__main__":
    # The order of execution matters due to dependencies
    train_classifier_if_needed()
    train_calibrator_if_needed()
    train_analogy_reasoner_if_needed()
    train_meta_reasoner_if_needed()