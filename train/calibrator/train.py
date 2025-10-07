# train/calibrator/train.py

import os
import joblib
from sklearn.linear_model import LogisticRegression
from src.logger import logger
from . import config, data_generator

def run_training():
    """
    Executes the full pipeline to train and save the confidence calibrator.
    """
    logger.info("--- Starting Confidence Calibrator Training ---")
    
    # 1. Generate the meta-dataset
    # X_cal = confidence scores, y_cal = 1 if correct, 0 if incorrect
    X_cal, y_cal = data_generator.generate_calibration_data()
    
    if X_cal is None:
        logger.error("Training stopped because calibration data could not be generated.")
        return
        
    # 2. Train the Logistic Regression model
    logger.info("Training the Logistic Regression calibrator...")
    calibrator = LogisticRegression()
    # We reshape X_cal because it's a single feature
    calibrator.fit(X_cal.reshape(-1, 1), y_cal)
    
    # 3. Save the trained calibrator
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_NAME)
    
    joblib.dump(calibrator, model_path)
    logger.info(f"✅ Confidence Calibrator saved successfully to {model_path}")
    logger.info("--- Training Finished ---")

if __name__ == '__main__':
    run_training()