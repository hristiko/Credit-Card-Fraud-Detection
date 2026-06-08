import time
import os
import numpy as np
from utils.preprocessing import load_data, preprocess_data, oversample_minority
from models.perceptron import Perceptron
from config import PERCEPTRON_EPOCHS, PERCEPTRON_LR, PERCEPTRON_WEIGHTS_PATH, PERCEPTRON_BIAS_PATH
from utils.metrics import compute_metrics, print_metrics

def main():
    print("\n----------------------------")
    print("Perceptron trianing")

    data_set = load_data("data/creditcard.csv")

    X_train, X_test, y_train, y_test = preprocess_data(data_set)

    X_balanced_train, y_balanced_train = oversample_minority(X_train, y_train)

    perceptron = Perceptron(input_dim=X_balanced_train.shape[1], lr=PERCEPTRON_LR)
    start_time = time.time()
    perceptron.train(X_balanced_train, y_balanced_train, epochs=PERCEPTRON_EPOCHS)
    end_time = time.time() - start_time

    os.makedirs(os.path.dirname(PERCEPTRON_WEIGHTS_PATH), exist_ok=True)
    np.save(PERCEPTRON_WEIGHTS_PATH, perceptron.weights)
    np.save(PERCEPTRON_BIAS_PATH, np.array(perceptron.bias))

    y_predicted = perceptron.predict(X_test)
    
    metrics = compute_metrics(y_correct_labels=y_test, y_predicted=y_predicted, evaluation_time=end_time)
    print_metrics(metrics, model_name="Perceptron")

if __name__ == "__main__":
    main()