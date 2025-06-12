# src/data_manager_bert.py
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import logger

class DataManagerBERT:
    def __init__(self, config: dict):
        """
        Initializes the DataManager for BERT embeddings.
        """
        self.config = config
        self.data_paths = self.config['data_paths']
        
        # Load pre-computed BERT embeddings
        self.bert_embeddings_path = Path("artifacts/banner_bert_embeddings.npy")
        self._load_resources()
        
    def _load_resources(self):
        """Loads raw data files and pre-computed embeddings."""
        logger.info("Loading raw data and BERT embeddings...")
        try:
            # Load BERT embeddings
            self.banner_embeddings = np.load(self.bert_embeddings_path)
            
            # Load training and testing dataframes
            self.p_train_df = pd.read_csv(self.data_paths['p_train'], index_col=0)
            self.p_test_df = pd.read_csv(self.data_paths['p_test'], index_col=0)
            
            logger.info("✅ Raw data and BERT embeddings loaded successfully.")
            logger.info(f"Shape of banner embeddings: {self.banner_embeddings.shape}")
        except FileNotFoundError as e:
            logger.error(f"❌ Error loading resources: {e}. Did you run generate_bert_embeddings.py?")
            raise

    def get_data_for_model(self, df: pd.DataFrame):
        """
        Prepares the final data for the model using BERT embeddings.
        """
        # 1. Match banner vectors to the dataframe using index 'j'
        X_banner_vectors = self.banner_embeddings[df['j'].values]
        
        # 2. Get segment indices 'i'
        X_segment_indices = df['i'].values
        
        X = [X_segment_indices, X_banner_vectors]
        
        # 3. Get target 'p' if it exists
        if 'p' in df.columns:
            y = df['p'].values
            return X, y
        else:
            return X, None