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

def create_model(config: dict, banner_input_shape: tuple) -> Model:
    """
    Creates and compiles the Keras model based on the NCF architecture.
    """
    model_config = config['model_params']
    
    # ... (بقیه کد این تابع بدون تغییر باقی می‌ماند) ...
    
    #================= Input Layers =================
    segment_input = Input(shape=(1,), name="segment_input")
    banner_input = Input(shape=banner_input_shape, sparse=True, name="banner_input")

    #================= Segment Embedding Branch =================
    segment_embedding = Embedding(
        input_dim=model_config['num_segments'],
        output_dim=model_config['segment_embedding_dim'],
        name="segment_embedding"
    )(segment_input)
    segment_vector = Flatten(name="flatten_segment_embedding")(segment_embedding)
    
    #================= Banner Processing Branch =================
    banner_vector = Dense(
        units=model_config['segment_embedding_dim'],
        activation='relu', 
        name="banner_dense_projection"
    )(banner_input)

    #================= Concatenate Branches =================
    concatenated = Concatenate()([segment_vector, banner_vector])
    
    #================= Fully Connected Layers =================
    x = concatenated
    for units in model_config['dense_layers']:
        x = Dense(units, activation='relu')(x)
        x = Dropout(model_config['dropout_rate'])(x)
        
    #================= Output Layer =================
    output = Dense(1, activation='linear', name='preference_output')(x)
    
    #================= Build and Compile Model =================
    model = Model(inputs=[segment_input, banner_input], outputs=output)
    
    optimizer = Adam(learning_rate=model_config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='mean_absolute_error',
        metrics=['mean_absolute_error', r_squared] # <--- ✨تغییر اینجاست
    )
    
    logger.info("✅ Keras model created and compiled successfully.")
    model.summary(print_fn=logger.info)
    
    return model