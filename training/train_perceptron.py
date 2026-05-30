import time
import os
import numpy as np
from utils.preprocessing import load_data, preprocess_data, save_scaler, oversample_minoroty
from models.perceptron import Perceptron
from config import PERCEPTRON_EPOCHS, PERCEPTRON_LR, SCALER_PATH, PERCEPTRON_WEIGHTS_PATH, PERCEPTRON_BIAS_PATH

def main():
    print("\n----------------------------")
    print("Perceptron trianing")

    data_set = load_data("data\creditcard.csv")

    X_train, X_test, y_train, y_test, save_scaler = preprocess_data(data_set)
    save_scaler(save_scaler, SCALER_PATH)

    X_balanced_train, y_balanced_train = oversample_minoroty(X_train, y_train)

    perceptron = Perceptron(input_dim=X_balanced_train[1], lr=PERCEPTRON_LR)
    start_time = time.time()
    perceptron.train(X_balanced_train, y_balanced_train, epochs=PERCEPTRON_EPOCHS)
    end_time = time.time() - start_time

    os.makedirs(os.path.dirname(PERCEPTRON_WEIGHTS_PATH), exist_ok=True)
    np.save(PERCEPTRON_WEIGHTS_PATH, perceptron.weights)
    np.save(PERCEPTRON_BIAS_PATH, np.array(perceptron.bias))

    y_predicted = perceptron.predict(X_test)
    y_predict_fraud_scores = perceptron.predict_fraud_scores(X_test)

    #metrics

if __name__ == "__main__":
    main()