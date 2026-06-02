import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

def build_model(input_dimension):
    model = Sequential([
        
        Input(shape=(input_dimension,)),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation="relu"),

        Dense(1, activation="sigmoid"),
    
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
        loss = "binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )

    return model