# src/core_pipeline.py

import os
import pandas as pd
from src.logger import logger
from src import config as main_config
from src.core.problem_classifier import ProblemClassifier
from src.core.test_loader import load_test_data
from src.core.reasoner import Reasoner
from sentence_transformers import SentenceTransformer
import textwrap

class CorePipeline:
    """
    Orchestrates the full reasoning process by initializing components
    and running the inference workflow.
    """
    def __init__(self):
        self._display_banner()
        logger.info("Initializing core reasoning pipeline...")
        
        self.classifier = None
        self.reasoner = None
        self.test_data = None
        self.embedding_model = None
        
        self._setup_directories()
        
        try:
            # Load all components
            classifier_model_path = os.path.join(main_config.MODELS_DIR, "problem_classifier.pkl")
            self.classifier = ProblemClassifier(model_path=classifier_model_path)
            self.reasoner = Reasoner()
            self.embedding_model = SentenceTransformer(main_config.EMBEDDING_MODEL_NAME)
            self.test_data = load_test_data()

        except Exception as e:
            logger.error(f"Failed to initialize a core component: {e}", exc_info=True)
            
        logger.info("Core reasoning pipeline initialized.")

    def run(self):
        """
        Runs the full inference pipeline on the loaded test data.
        """
        if self.test_data is None or self.test_data.empty or self.classifier is None or self.reasoner is None:
            logger.error("A required component or data is not available. Aborting run.")
            return None

        logger.info("Starting inference run on test data...")
        
        results = []
        for index, row in self.test_data.iterrows():
            print(f"\n{'='*25} Processing Row {index+1} {'='*25}")
            
            print("\n[ PROBLEM STATEMENT ]")
            print(textwrap.fill(row['problem_statement'], width=80))
            print("\n[ OPTIONS ]")
            print(f"  1: {row['answer_option_1']}")
            print(f"  2: {row['answer_option_2']}")
            print(f"  3: {row['answer_option_3']}")
            print(f"  4: {row['answer_option_4']}")
            print(f"  5: {row['answer_option_5']}")
            print("-" * 65)
            
            embedding = self.embedding_model.encode(row['problem_statement'], convert_to_tensor=True)
            
            # 1. Classify Topic
            predicted_topic = self.classifier.predict(row=row, embedding=embedding, row_index=index)
            
            # 2. Get Symbolic Answer
            symbolic_result = self.reasoner.solve_symbolically(row=row, topic=predicted_topic)
            if symbolic_result:
                print(f"-> Symbolic Output: option number: {symbolic_result['answer']} - confidence: {symbolic_result['confidence']}")
            
            # 3. Get Heuristic (LLM) Answer
            heuristic_result = self.reasoner.solve_heuristically(row=row, topic=predicted_topic)
            if heuristic_result:
                solution_text = textwrap.fill(heuristic_result['solution'], width=70, initial_indent="    ", subsequent_indent="    ")
                print(f"-> Heuristic Output: option_number: {heuristic_result['answer']}, solution: \n{solution_text}\n     , then confidence: {heuristic_result['confidence']}")

            results.append({
                'predicted_topic': predicted_topic,
                'symbolic_answer': symbolic_result['answer'] if symbolic_result else None,
                'symbolic_confidence': symbolic_result['confidence'] if symbolic_result else None,
                'heuristic_answer': heuristic_result['answer'] if heuristic_result else None,
                'heuristic_confidence': heuristic_result['confidence'] if heuristic_result else None,
                'heuristic_solution': heuristic_result['solution'] if heuristic_result else None
            })

        result_df = pd.DataFrame(results)
        self.test_data = pd.concat([self.test_data, result_df], axis=1)
        
        logger.info("Inference run complete.")
        
        print("\n\n" + "="*30 + " FINAL RESULTS PREVIEW " + "="*30)
        print(self.test_data[[
            'problem_statement', 'predicted_topic', 'symbolic_answer', 'heuristic_answer', 'heuristic_solution'
        ]].head())
        print("="*85)
        
        return self.test_data


    def _display_banner(self):
        """Displays a welcome banner for the application."""
        banner = f"""
        ====================================================
            🧠 Agentic Reasoning Engine - Version {main_config.VERSION}
        ====================================================
        """
        print(banner)

    def _setup_directories(self):
        """Creates necessary output directories if they don't exist."""
        logger.info("Setting up output directories...")
        os.makedirs(main_config.OUTPUT_JSON_DIR, exist_ok=True)
        os.makedirs(main_config.OUTPUT_CSV_DIR, exist_ok=True)
        logger.info("Output directories are ready.")