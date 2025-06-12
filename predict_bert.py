# predict_bert.py
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model

# Import project modules
from src.config import config
from src.data_manager_bert import DataManagerBERT          # <--- تغییر ۱: استفاده از دیتا منیجر جدید
from src.model_pipeline_bert import r_squared             # <--- تغییر ۲: وارد کردن متریک سفارشی
from src.utils import logger

def run_bert_prediction():
    """
    Executes the prediction pipeline for the BERT-based model.
    """
    logger.info("="*20 + " Starting BERT Prediction Pipeline " + "="*20)

    # --- ۱. تعریف مسیر و بارگذاری مدل BERT ---
    artifacts_dir = Path(config['artifact_paths']['artifacts_dir'])
    # ✨ استفاده از نام فایل مدل جدید
    model_path = artifacts_dir / "preference_model_bert.h5"

    logger.info(f"Loading saved BERT model from {model_path}...")
    try:
        # ✨ هنگام بارگذاری مدل با متریک سفارشی، باید آن را به Keras معرفی کنیم
        model = load_model(model_path, custom_objects={'r_squared': r_squared})
        logger.info("✅ BERT Model loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load BERT model: {e}. Please run `train_bert.py` first.")
        return

    # --- ۲. آماده‌سازی داده‌های تست با DataManagerBERT ---
    # ✨ استفاده از دیتا منیجر جدید
    data_manager = DataManagerBERT(config=config)

    logger.info("Preparing test data for prediction using BERT embeddings...")
    # ✨ دیگر نیازی به ارسال وکتورایزر نیست
    X_test, _ = data_manager.get_data_for_model(
        df=data_manager.p_test_df
    )
    logger.info(f"Test data prepared. Input shapes: {[x.shape for x in X_test]}")

    # --- ۳. تولید پیش‌بینی (بدون تغییر) ---
    logger.info("🚀 Generating predictions...")
    predictions = model.predict(X_test)
    predictions = predictions.flatten()
    logger.info(f"✅ Predictions generated successfully. Shape: {predictions.shape}")

    # --- ۴. ساخت و ذخیره فایل ارسال ---
    submission_df = pd.DataFrame({'p': predictions})
    # ✨ برای جلوگیری از رونویسی، فایل را با نام جدیدی ذخیره می‌کنیم
    submission_path = Path("submission_bert.csv")

    logger.info(f"Saving submission file to {submission_path}...")
    submission_df.to_csv(submission_path, index=False)

    logger.info(f"🏆 BERT submission file created successfully at: {submission_path}")
    logger.info("="*20 + " BERT Prediction Pipeline Finished " + "="*20)


if __name__ == '__main__':
    run_bert_prediction()