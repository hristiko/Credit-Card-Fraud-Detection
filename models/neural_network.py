import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation

def build_model(input_dimension):
    model = Sequential([
        
        Input(shape=(input_dimension,)),

        Dense(64, kernel_initializer="he_normal"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3),

        Dense(32, kernel_initializer="he_normal"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),

        Dense(16, kernel_initializer="he_normal"),
        BatchNormalization(),
        Activation("relu"),

        Dense(1, activation="sigmoid"),
    
    ])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate = 0.001, weight_decay = 0.0001),
        loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, apply_class_balancing=True),
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="pr_auc", curve="PR")]
    )

    return model