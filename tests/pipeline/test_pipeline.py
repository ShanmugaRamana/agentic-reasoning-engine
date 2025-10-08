import sys
import os
import pandas as pd

# Add project root to path to allow importing from src
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.core_pipeline import CorePipeline
from src.logger import logger

# A class to redirect output to both console and file
class Tee:
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stream.flush()

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
    Initializes the pipeline and runs all test functions, capturing output.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    with open('test.log', 'w', encoding='utf-8') as log_file:
        tee = Tee(log_file, original_stdout)
        sys.stdout = tee
        sys.stderr = tee
        
        try:
            logger.info("=============================================")
            logger.info("          STARTING PIPELINE TESTS            ")
            logger.info("=============================================")
            
            pipeline = CorePipeline()
            
            init_ok = test_pipeline_initialization(pipeline)
            
            if init_ok:
                # --- THIS LINE IS NOW CORRECTED ---
                test_pipeline_run_method(pipeline)
            
            logger.info("=============================================")
            logger.info("        TESTING COMPLETE - See test.log      ")
            logger.info("=============================================")
            
        except Exception as e:
            logger.error(f"A critical error occurred during testing: {e}", exc_info=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    run_all_tests()