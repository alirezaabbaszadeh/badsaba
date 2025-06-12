# src/model_pipeline_bert.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import AdamW
from src.utils import logger
from src.model_pipeline import r_squared # میتوانیم از همان تابع r_squared قبلی استفاده کنیم

# ✨ تغییر ۱: افزودن پارامتر برای تعداد ویژگی‌های جدید به تابع
def create_bert_model(config: dict, num_extra_features: int) -> Model:
    model_config = config['model_params']
    
    # --- ورودی‌ها ---
    segment_input = Input(shape=(1,), name="segment_input")
    banner_input = Input(shape=(768,), name="banner_bert_input")
    # ✨ تغییر ۲: تعریف ورودی سوم برای ویژگی‌های مهندسی‌شده
    extra_features_input = Input(shape=(num_extra_features,), name="extra_features_input")

    # --- شاخه سگمنت (بدون تغییر) ---
    segment_embedding = Embedding(
        input_dim=model_config['num_segments'],
        output_dim=model_config['segment_embedding_dim'],
        name="segment_embedding"
    )(segment_input)
    segment_vector = Flatten(name="flatten_segment_embedding")(segment_embedding)
    
    # --- شاخه بنر (بدون تغییر) ---
    banner_vector = Dense(
        units=model_config['segment_embedding_dim'], 
        activation='relu', 
        name="banner_dense_projection"
    )(banner_input)

    # ✨ تغییر ۳: ادغام هر سه ورودی با هم
    # توجه: بردار ویژگی‌های اضافی را مستقیماً به لایه ادغام متصل می‌کنیم
    concatenated = Concatenate()([segment_vector, banner_vector, extra_features_input])
    
    x = concatenated
    for units in model_config['dense_layers']:
        x = Dense(units, activation='relu')(x)
        x = Dropout(model_config['dropout_rate'])(x)
        
    output = Dense(1, activation='linear', name='preference_output')(x)
    
    # ✨ تغییر ۴: به‌روزرسانی لیست ورودی‌های مدل
    model = Model(inputs=[segment_input, banner_input, extra_features_input], outputs=output)
    
    optimizer = AdamW(learning_rate=model_config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error', # می‌توانید اینجا 'huber_loss' را هم امتحان کنید
        metrics=['mean_absolute_error', r_squared]
    )
    
    logger.info("✅ Keras BERT model with extra features created and compiled successfully.")
    model.summary(print_fn=logger.info)
    
    return model