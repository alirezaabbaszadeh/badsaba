# train_bert.py
from pathlib import Path
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ✨ ماژول‌های جدید را وارد می‌کنیم
from src.config import config
from src.data_manager_bert import DataManagerBERT 
from src.model_pipeline_bert import create_bert_model
from src.utils import logger

from pathlib import Path
import numpy as np
# ✨ ۱. ReduceLROnPlateau را از keras.callbacks وارد کنید
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.model_pipeline_bert import create_bert_model


def run_bert_training():
    logger.info("="*20 + " Starting BERT Model Training Pipeline " + "="*20)

    # 1. استفاده از DataManagerBERT
    data_manager = DataManagerBERT(config=config)

    # 2. آماده سازی داده‌ها (دیگر نیازی به ساخت وکتورایزر نیست)
    logger.info("Preparing training data for the BERT model...")
    X_train, y_train = data_manager.get_data_for_model(df=data_manager.p_train_df)
    
    # 3. ساخت مدل BERT
  


    # ✨ تغییر: محاسبه تعداد ویژگی‌های جدید و پاس دادن آن به تابع ساخت مدل
    num_extra_features = X_train[2].shape[1] # X_train[2] همان ویژگی‌های مهندسی‌شده است

    logger.info(f"Number of engineered features: {num_extra_features}")
    # --- ۴. ساخت مدل BERT ---
    # model = create_bert_model(config=config)
    model = create_bert_model(config=config, num_extra_features=num_extra_features)

    
    # --- ۵. تعریف Callbacks برای آموزش ---
    artifacts_dir = Path(config['artifact_paths']['artifacts_dir'])
    model_path = artifacts_dir / "preference_model_bert.h5" 
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    
    # ✨ ۲. تعریف کردن ReduceLROnPlateau
    # اگر خطای اعتبارسنجی (val_loss) بعد از 2 دور بهبود پیدا نکرد،
    # نرخ یادگیری را با ضریب 0.2 গুণ کن (یعنی 80% کاهش بده).
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,             # ضریب کاهش: new_lr = lr * factor
        patience=1,             # تعداد دورهای انتظار برای بهبود
        min_lr=0.00001,         # حداقل مقدار ممکن برای نرخ یادگیری
        verbose=1               # نمایش پیام هنگام کاهش نرخ یادگیری
    )
    
    # --- ۶. آموزش مدل ---
    logger.info("🚀 Starting BERT model training...")
    model_params = config['model_params']
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=model_params['batch_size'],
        epochs=model_params['epochs'],
        validation_split=model_params['validation_split'],
        # ✨ ۳. اضافه کردن callback جدید به لیست
        callbacks=[early_stopping, model_checkpoint, reduce_lr] 
    )


    # ... (بخش لاگ کردن نتایج نهایی مانند قبل)
    best_epoch_idx = np.argmin(history.history['val_loss'])
    best_val_r2 = history.history['val_r_squared'][best_epoch_idx]
    logger.info(f"🏆 Best BERT model saved to: {model_path}")
    logger.info(f"  -> Best Validation R-squared: {best_val_r2:.4f}")
    logger.info("="*20 + " BERT Model Training Pipeline Finished " + "="*20)

if __name__ == '__main__':
    run_bert_training()