# Script to load a trained model and generate predictions
# predict.py
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model

# Import project modules
from src.config import config
from src.data_manager import DataManager
from src.utils import logger, load_object

def run_prediction():
    """
    Executes the entire prediction pipeline to generate the submission file.
    """
    logger.info("="*20 + " Starting Prediction Pipeline " + "="*20)

    # --- 1. Define paths and load saved artifacts ---
    artifacts_dir = Path(config['artifact_paths']['artifacts_dir'])
    vectorizer_path = artifacts_dir / config['artifact_paths']['vectorizer_file']
    model_path = artifacts_dir / config['artifact_paths']['model_file']

    logger.info("Loading saved artifacts (model and vectorizer)...")
    try:
        # Load the TfidfVectorizer object from the file
        vectorizer = load_object(file_path=vectorizer_path)
        # Load the entire Keras model
        model = load_model(model_path)
        logger.info("✅ Artifacts loaded successfully.")
    except FileNotFoundError:
        logger.error(
            f"❌ Model or vectorizer not found in {artifacts_dir}. "
            "Please run the training pipeline first by executing `python train.py`."
        )
        return  # Exit the script if artifacts are missing



# predict.py

# ... (کدهای قبلی فایل)

    # --- 2. Initialize DataManager and prepare test data ---
    data_manager = DataManager(config=config)
    
    logger.info("Preparing test data for prediction...")
    
    # ✨ تغییر کلیدی: دریافت دو مقدار و نادیده گرفتن مقدار دوم (y)
    X_test, _ = data_manager.get_data_for_model(
        df=data_manager.p_test_df,
        vectorizer=vectorizer
    )
    
    logger.info(f"Test data prepared. Input shapes: {[x.shape for x in X_test]}")

# ... (بقیه کدهای فایل)





    # --- 3. Generate Predictions ---
    logger.info("🚀 Generating predictions on the test set...")
    predictions = model.predict(X_test)
    
    # The model output is a 2D array of shape (n_samples, 1).
    # We flatten it to get a 1D array for the CSV column.
    predictions = predictions.flatten()
    logger.info(f"✅ Predictions generated. Total predictions: {len(predictions)}")

    # --- 4. Create and Save Submission File ---
    # Create a pandas DataFrame with a single column 'p' as required.
    submission_df = pd.DataFrame({'p': predictions})
    
    submission_path = Path(config['data_paths']['submission_file'])
    
    logger.info(f"Saving submission file to: {submission_path}")
    # Save to CSV without the pandas index.
    submission_df.to_csv(submission_path, index=False)
    
    logger.info(f"🏆 Submission file created successfully!")
    logger.info("="*20 + " Prediction Pipeline Finished " + "="*20)


if __name__ == '__main__':
    run_prediction()