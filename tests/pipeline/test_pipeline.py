import sys
import os
import pandas as pd

# Add project root to path to allow importing from src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.core_pipeline import CorePipeline
from src.logger import logger

def test_pipeline_initialization(pipeline: CorePipeline):
    """
    Tests if all components of the CorePipeline are loaded correctly.
    """
    logger.info("--- 🧪 Running Test: Pipeline Initialization ---")
    
    if pipeline is None:
        logger.error("❌ FAIL: Pipeline object is None.")
        return False
    
    if pipeline.classifier is None or pipeline.classifier.model is None:
        logger.error("❌ FAIL: Classifier component did not initialize correctly.")
        return False
        
    if pipeline.test_data is None or not isinstance(pipeline.test_data, pd.DataFrame) or pipeline.test_data.empty:
        logger.error("❌ FAIL: Test data did not load correctly.")
        return False

    logger.info("✅ PASS: Pipeline initialization successful.")
    return True

def test_pipeline_run_method(pipeline: CorePipeline):
    """
    Tests the main run() method of the pipeline.
    """
    logger.info("--- 🧪 Running Test: Pipeline Run Method ---")
    
    # Execute the main pipeline logic
    pipeline.run()

    if 'predicted_topic' not in pipeline.test_data.columns:
        logger.error("❌ FAIL: The 'predicted_topic' column was not added to the test data.")
        return False

    if not pipeline.test_data['predicted_topic'].notna().all():
        logger.error("❌ FAIL: The 'predicted_topic' column contains null values.")
        return False
        
    logger.info("✅ PASS: Pipeline run() method executed successfully.")
    return True

def run_all_tests():
    """
    Initializes the pipeline and runs all test functions.
    """
    logger.info("=============================================")
    logger.info("          STARTING PIPELINE TESTS            ")
    logger.info("=============================================")
    
    try:
        pipeline = CorePipeline()
        
        # Run tests in sequence
        init_ok = test_pipeline_initialization(pipeline)
        
        # Only run the second test if the first one passed
        if init_ok:
            run_ok = test_pipeline_run_method(pipeline)
        
        logger.info("=============================================")
        logger.info("             TESTING COMPLETE                ")
        logger.info("=============================================")
        
    except Exception as e:
        logger.error(f"A critical error occurred during testing: {e}", exc_info=True)

if __name__ == "__main__":
    run_all_tests()