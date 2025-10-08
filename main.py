# main.py

from scripts.process_data import run_pipeline
from scripts.train_components import (
    train_classifier_if_needed,
    train_analogy_reasoner_if_needed,
    train_calibrator_if_needed,
    train_meta_reasoner_if_needed
)
from scripts.run_inference import run_inference
from src.logger import logger

if __name__ == "__main__":
    
    logger.info("=============================================")
    logger.info("          STARTING SAGE-V1 ENGINE            ")
    logger.info("=============================================")

    # --- BUILD STAGE ---
    logger.info("--- STAGE 1: Verifying System Build ---")
    run_pipeline()
    train_classifier_if_needed()
    train_calibrator_if_needed()
    train_analogy_reasoner_if_needed()
    train_meta_reasoner_if_needed()
    logger.info("--- System Build Verified ---")

    # --- INFERENCE STAGE ---
    logger.info("--- STAGE 2: Starting Inference Run ---")
    run_inference()