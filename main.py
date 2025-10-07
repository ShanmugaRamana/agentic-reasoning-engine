# main.py

from scripts.process_data import run_pipeline
from scripts.train_components import (
    train_classifier_if_needed, 
    train_analogy_reasoner_if_needed,
    train_calibrator_if_needed,
    train_meta_reasoner_if_needed
)

if __name__ == "__main__":
    """
    This is the main entry point for the application.
    It orchestrates the entire pipeline:
    1. Processes raw data into a usable format.
    2. Trains all ML components in the correct order if they don't exist.
    """
    # Step 1: Process the raw data (skipped if already done)
    run_pipeline()
    
    # Step 2: Train the problem classifier (skipped if model exists)
    train_classifier_if_needed()

    # Step 3: Train the confidence calibrator (depends on the classifier)
    train_calibrator_if_needed()

    # Step 4: Train the analogical reasoner (skipped if model exists)
    train_analogy_reasoner_if_needed()

    # Step 5: Train the meta-reasoner (depends on the analogy reasoner)
    train_meta_reasoner_if_needed()