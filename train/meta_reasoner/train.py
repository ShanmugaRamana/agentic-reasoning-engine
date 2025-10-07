# train/meta_reasoner/train.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.logger import logger
from . import config, feature_generator

def run_training():
    """
    Executes the full pipeline to train and save the meta-reasoner.
    """
    logger.info("--- Starting Meta-Reasoner Training ---")
    
    # 1. Generate the meta-dataset
    X_meta, y_meta = feature_generator.generate_meta_features()
    
    if X_meta is None:
        logger.error("Training stopped because meta-features could not be generated.")
        return
        
    # 2. Encode labels ('ANALOGY', 'LLM') into integers
    label_encoder = LabelEncoder()
    y_meta_encoded = label_encoder.fit_transform(y_meta)
    
    # 3. Train the RandomForestClassifier
    logger.info("Training the RandomForestClassifier as the meta-reasoner...")
    meta_reasoner = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    meta_reasoner.fit(X_meta, y_meta_encoded)
    
    # 4. Create the artifact to be saved
    artifact = {
        'meta_reasoner': meta_reasoner,
        'label_encoder': label_encoder
    }
    
    # 5. Save the trained meta-reasoner
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, config.MODEL_NAME)
    
    joblib.dump(artifact, model_path)
    logger.info(f"✅ Meta-Reasoner saved successfully to {model_path}")
    logger.info("--- Training Finished ---")

if __name__ == '__main__':
    run_training()