# Handles data loading and preprocessing pipelines
# src/data_manager.py
from pathlib import Path
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from hazm import Normalizer
from src.config import config
from src.utils import logger, save_object

class DataManager:
    def __init__(self, config: dict):
        """
        Initializes the DataManager with paths and parameters from the config.
        """
        self.config = config
        self.data_paths = self.config['data_paths']
        self.feature_params = self.config['feature_params']['tfidf']

        # Initialize Persian text normalizer
        self.normalizer = Normalizer()
        
        # Load all raw data into memory
        self._load_raw_data()

    def _load_raw_data(self):
        """Loads raw data files into pandas DataFrames."""
        logger.info("Loading raw data...")
        try:
            # Load banner captions
            with open(self.data_paths['banners'], 'r', encoding='utf-8') as f:
                banner_captions = [line.strip() for line in f.readlines()]
            self.banners_df = pd.DataFrame(banner_captions, columns=['caption'])

            # Load training and testing data
            self.p_train_df = pd.read_csv(self.data_paths['p_train'], index_col=0)
            self.p_test_df = pd.read_csv(self.data_paths['p_test'], index_col=0)
            logger.info("✅ Raw data loaded successfully.")
            logger.info(f"Train data shape: {self.p_train_df.shape}")
            logger.info(f"Test data shape: {self.p_test_df.shape}")
            logger.info(f"Banners count: {len(self.banners_df)}")
        except FileNotFoundError as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """Normalizes Persian text."""
        return self.normalizer.normalize(text)

    def build_vectorizer(self, save_path: Path):
        """
        Builds and fits a TF-IDF vectorizer on the banner captions.
        Saves the fitted vectorizer to a file.
        """
        logger.info("Building and fitting TF-IDF vectorizer...")
        
        # Preprocess captions
        preprocessed_captions = self.banners_df['caption'].apply(self._preprocess_text)
        
        # Initialize and fit vectorizer
        vectorizer = TfidfVectorizer(
            max_features=self.feature_params['max_features'],
            ngram_range=tuple(self.feature_params['ngram_range'])
        )
        vectorizer.fit(preprocessed_captions)
        
        # Save the fitted vectorizer
        save_object(save_path, vectorizer)
        logger.info(f"✅ Vectorizer built and saved to {save_path}")
        
        return vectorizer





# src/data_manager.py

    def get_data_for_model(self, df: pd.DataFrame, vectorizer: TfidfVectorizer):
        """
        Prepares the final data for the model by combining segment and banner features.
        
        Returns:
            A tuple of (X, y) where X is the list of inputs for the model and
            y is the target values. For test data, y is None.
        """
        # 1. Get banner vectors using their index 'j'
        preprocessed_captions = self.banners_df['caption'].apply(self._preprocess_text)
        banner_vectors = vectorizer.transform(preprocessed_captions)
        
        # Match banner vectors to the train/test set based on 'j' index
        X_banner_vectors = banner_vectors[df['j']]
        
        # 2. Get segment indices 'i'
        X_segment_indices = df['i'].values

        # Define the input list for the model
        X = [X_segment_indices, X_banner_vectors]
        
        # 3. Get target 'p' if it exists, otherwise return None for y
        if 'p' in df.columns:
            y = df['p'].values
            return X, y
        else:
            return X, None # ✨ تغییر کلیدی: همیشه دو مقدار برگردانده می‌شود