# src/data_manager_bert.py
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import logger

class DataManagerBERT:
    def __init__(self, config: dict):
        """
        Initializes the DataManager for BERT embeddings and new features.
        """
        self.config = config
        # ✨ تغییر ۱: تعریف مسیر داده‌های پردازش‌شده
        self.processed_data_path = Path("data/processed")
        
        # Load pre-computed BERT embeddings
        self.bert_embeddings_path = Path("artifacts/banner_bert_embeddings.npy")
        self._load_resources()
        
    def _load_resources(self):
        """Loads processed data files and pre-computed embeddings."""
        logger.info("Loading processed data and BERT embeddings...")
        try:
            # Load BERT embeddings
            self.banner_embeddings = np.load(self.bert_embeddings_path)
            
            # ✨ تغییر ۲: بارگذاری دیتافریم‌های دارای ویژگی‌های جدید
            train_path = self.processed_data_path / "train_featured.csv"
            test_path = self.processed_data_path / "test_featured.csv"
            
            self.p_train_df = pd.read_csv(train_path, index_col=0)
            self.p_test_df = pd.read_csv(test_path, index_col=0)
            
            logger.info("✅ Processed data and BERT embeddings loaded successfully.")
            logger.info(f"Shape of banner embeddings: {self.banner_embeddings.shape}")
            logger.info(f"Train dataframe shape: {self.p_train_df.shape}")
            logger.info(f"Test dataframe shape: {self.p_test_df.shape}")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Error loading resources: {e}. Did you run build_features.py?")
            raise

    def get_data_for_model(self, df: pd.DataFrame):
        """
        Prepares the final data for the model using BERT embeddings and engineered features.
        """
        # ۱. استخراج بردارهای بنر (بدون تغییر)
        X_banner_vectors = self.banner_embeddings[df['j'].values]
        
        # ۲. استخراج شناسه‌های سگمنت (بدون تغییر)
        X_segment_indices = df['i'].values
        
        # ✨ تغییر ۳: استخراج ویژگی‌های مهندسی‌شده جدید
        feature_columns = [
            'text_length', 'word_count', 'banner_avg_score', 'banner_view_count',
            'banner_score_std', 'segment_avg_score', 'segment_interaction_count',
            'segment_score_std'
        ]
        # اطمینان از اینکه همه ستون‌ها در دیتافریم وجود دارند
        existing_feature_columns = [col for col in feature_columns if col in df.columns]
        X_extra_features = df[existing_feature_columns].values
        
        # لیست ورودی‌های مدل اکنون شامل ۳ بخش است
        X = [X_segment_indices, X_banner_vectors, X_extra_features]
        
        # ۴. استخراج متغیر هدف (بدون تغییر)
        if 'p' in df.columns:
            y = df['p'].values
            return X, y
        else:
            return X, None