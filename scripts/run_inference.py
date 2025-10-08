# scripts/run_inference.py

import sys
import os
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core_pipeline import CorePipeline
from src.logger import logger
from src import config as main_config

def run_inference():
    """
    Initializes and runs the main inference pipeline, then saves the results.
    Skips the run if output files already exist.
    """
    logger.info("--- Starting Inference Run Script ---")
    
    # --- NEW: Check if output files already exist ---
    if os.path.exists(main_config.OUTPUT_CSV_PATH) and os.path.exists(main_config.OUTPUT_JSON_PATH):
        logger.info("Output files (output.csv and output.json) already exist. Skipping inference run.")
        logger.info("--- Inference Run Finished ---")
        return

    try:
        pipeline = CorePipeline()
        # The run method now returns the results DataFrame
        results_df = pipeline.run()

        if results_df is not None and not results_df.empty:
            # --- NEW: Prepare and save the output files ---
            logger.info("Preparing final output files...")
            
            # Select and rename columns as per requirements
            output_df = results_df[[
                'predicted_topic', 
                'problem_statement', 
                'heuristic_solution', 
                'heuristic_answer'
            ]].rename(columns={
                'predicted_topic': 'topic',
                'heuristic_solution': 'solution',
                'heuristic_answer': 'correct_option'
            })

            # Save to CSV
            output_df.to_csv(main_config.OUTPUT_CSV_PATH, index=False)
            logger.info(f"Successfully saved output to {main_config.OUTPUT_CSV_PATH}")
            
            # Save to JSON
            output_df.to_json(main_config.OUTPUT_JSON_PATH, orient='records', indent=4)
            logger.info(f"Successfully saved output to {main_config.OUTPUT_JSON_PATH}")
        else:
            logger.error("Pipeline run did not produce any results to save.")

        logger.info("--- Inference Run Finished Successfully ---")

    except Exception as e:
        logger.error(f"A critical error occurred during the inference run: {e}", exc_info=True)

if __name__ == '__main__':
    run_inference()