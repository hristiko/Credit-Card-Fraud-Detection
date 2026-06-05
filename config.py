import os

RANDOM_SEED = 42
TEST_SIZE = 0.2 

#Perceptron configuration
PERCEPTRON_LR = 0.01
PERCEPTRON_EPOCHS = 5
SCALER_PATH = "saved_models/scaler.pkl"
PERCEPTRON_WEIGHTS_PATH = "saved_models/perceptron_weights.npy"
PERCEPTRON_BIAS_PATH = "saved_models/perceptron_bias.npy"

#NN configuration
NN_EPOCHS = 30
NN_BATCH_SIZE = 512
NN_MODEL_PATH = "saved_models/nn_model.keras"
NN_THRESHOLD_PATH = "saved_models/nn_threshold.txt"

os.makedirs("saved_models", exist_ok=True)