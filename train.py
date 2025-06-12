# train.py
from pathlib import Path
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import project modules
from src.config import config
from src.data_manager import DataManager
from src.model_pipeline import create_model
from src.utils import logger

def run_training():
    """
    Executes the entire model training pipeline.
    """
    logger.info("="*20 + " Starting Model Training Pipeline " + "="*20)

    # ... (بخش‌های ۱ تا ۵ بدون تغییر باقی می‌مانند) ...
    # --- 1. Initialize DataManager ---
    data_manager = DataManager(config=config)
    # --- 2. Build and Save the Vectorizer ---
    artifacts_dir = Path(config['artifact_paths']['artifacts_dir'])
    vectorizer_path = artifacts_dir / config['artifact_paths']['vectorizer_file']
    vectorizer = data_manager.build_vectorizer(save_path=vectorizer_path)
    # --- 3. Prepare Data for the Model ---
    logger.info("Preparing training data for the model...")
    X_train, y_train = data_manager.get_data_for_model(df=data_manager.p_train_df, vectorizer=vectorizer)
    logger.info(f"Training data prepared. Input shapes: {[x.shape for x in X_train]}")
    # --- 4. Create and Compile the Keras Model ---
    banner_input_shape = (X_train[1].shape[1],)
    model = create_model(config=config, banner_input_shape=banner_input_shape)
    # --- 5. Define Callbacks for Training ---
    model_params = config['model_params']
    model_path = artifacts_dir / config['artifact_paths']['model_file']
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    
    # --- 6. Train the Model ---
    logger.info("🚀 Starting model training...")
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=model_params['batch_size'],
        epochs=model_params['epochs'],
        validation_split=model_params['validation_split'],
        callbacks=[early_stopping, model_checkpoint]
    )

    logger.info("✅ Model training completed.")
    
    # --- 7. Log Final Model Performance ---  # <--- ✨بخش جدید اینجاست
    logger.info("="*20 + " Final Model Performance " + "="*20)
    
    # Find the best epoch based on validation loss
    best_epoch_idx = np.argmin(history.history['val_loss'])
    
    # Get the metrics from the best epoch
    best_val_loss = history.history['val_loss'][best_epoch_idx]
    best_val_mae = history.history['val_mean_absolute_error'][best_epoch_idx]
    best_val_r2 = history.history['val_r_squared'][best_epoch_idx]
    
    logger.info(f"🏆 Best model saved to: {model_path}")
    logger.info(f"📈 Best Epoch: {best_epoch_idx + 1}")
    logger.info(f"  -> Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"  -> Best Validation MAE: {best_val_mae:.4f}")
    logger.info(f"  -> Best Validation R-squared: {best_val_r2:.4f}")

    logger.info("="*20 + " Model Training Pipeline Finished " + "="*20)

if __name__ == '__main__':
    run_training()