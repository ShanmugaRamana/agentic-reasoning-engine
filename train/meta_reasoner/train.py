# train/meta_reasoner/train.py
import os
import joblib
from src.logger import logger
from . import config, performance_evaluator, weight_optimizer

def run_training():
    """
    Executes the full pipeline to build and save the meta-reasoner weights.
    """
    logger.info("--- Starting Meta-Reasoner Weight Generation ---")
    
    # 1. Evaluate the performance of each reasoner
    performance_data = performance_evaluator.evaluate_reasoner_performance()
    
    # 2. Optimize/Normalize the performance scores into weights
    final_weights = weight_optimizer.optimize_weights(performance_data)
    
    # 3. The "model" artifact is the dictionary of weights
    artifact = final_weights
    
    # 4. Save the artifact
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_NAME)
    
    joblib.dump(artifact, model_path)
    logger.info(f"✅ Meta-Reasoner weights saved successfully to {model_path}")
    logger.info("--- Meta-Reasoner Generation Finished ---")

if __name__ == '__main__':
    run_training()