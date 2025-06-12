# src/model_pipeline.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Dense, Concatenate, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K # Import Keras backend
from src.utils import logger

def r_squared(y_true, y_pred):
    """
    Custom R-squared metric for Keras.
    """
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))





# src/model_pipeline_bert.py
# ... (کپی کردن تمام import ها از model_pipeline.py) ...
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import AdamW
from src.utils import logger
from src.model_pipeline import r_squared # میتوانیم از همان تابع r_squared قبلی استفاده کنیم

def create_bert_model(config: dict) -> Model:
    model_config = config['model_params']
    
    # --- ورودی‌ها ---
    segment_input = Input(shape=(1,), name="segment_input")
    # ✨ تغییر: ورودی دیگر اسپارس نیست و ابعاد آن 768 است
    banner_input = Input(shape=(768,), name="banner_bert_input")

    # --- شاخه سگمنت (بدون تغییر) ---
    segment_embedding = Embedding(
        input_dim=model_config['num_segments'],
        output_dim=model_config['segment_embedding_dim'],
        name="segment_embedding"
    )(segment_input)
    segment_vector = Flatten(name="flatten_segment_embedding")(segment_embedding)
    
    # --- شاخه بنر (حالا ورودی BERT را می‌گیرد) ---
    # از آنجایی که بردارهای BERT خودشان معنایی هستند، می‌توانیم آنها را مستقیماً استفاده کنیم
    # یا از یک لایه Dense برای تنظیم ابعاد استفاده کنیم. ما گزینه دوم را انتخاب می‌کنیم.
    banner_vector = Dense(
        units=model_config['segment_embedding_dim'], 
        activation='relu', 
        name="banner_dense_projection"
    )(banner_input)

    # --- ترکیب و ادامه (بدون تغییر) ---
    concatenated = Concatenate()([segment_vector, banner_vector])
    
    x = concatenated
    for units in model_config['dense_layers']:
        x = Dense(units, activation='relu')(x)
        x = Dropout(model_config['dropout_rate'])(x)
        
    output = Dense(1, activation='linear', name='preference_output')(x)
    
    model = Model(inputs=[segment_input, banner_input], outputs=output)
    
    optimizer = AdamW(learning_rate=model_config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error',
        metrics=['mean_absolute_error', r_squared]
    )
    
    logger.info("✅ Keras BERT model created and compiled successfully.")
    model.summary(print_fn=logger.info)
    
    return model