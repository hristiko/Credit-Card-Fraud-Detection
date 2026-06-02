import numpy as np
import os
import numpy as np
import pickle
import tensorflow as tf
from utils.preprocessing import load_data, preprocess_data
from utils.metrics import compute_metrics, print_metrics
from models.perceptron import Perceptron
from config import PERCEPTRON_WEIGHTS_PATH, PERCEPTRON_BIAS_PATH, NN_MODEL_PATH, NN_THRESHOLD_PATH

def load_test_data():

    dataset = load_data("data/creditcard.csv")

    X_train, X_test, y_train, y_test, _ = preprocess_data(dataset)

    return X_test, y_test

def evaluate_perceptron(X_test, y_test):

    print("\nEvaluation on perceptron started . . .")

    weights = np.load(PERCEPTRON_WEIGHTS_PATH)
    bias = float(np.load(PERCEPTRON_BIAS_PATH))

    model = Perceptron(input_dim=X_test.shape[1])
    model.weights = weights
    model.bias = bias

    y_predicted = model.predict(X_test)

    return compute_metrics(y_correct_labels = y_test, y_predicted = y_predicted)

def load_nn_threshold():

    if os.path.exists(NN_THRESHOLD_PATH):
        with open(NN_THRESHOLD_PATH, "r") as f:
            threshold = float(f.read().strip())
        return threshold
    else:
        print("Theshold not found. The defaul (0.5) will be used")
        threshold = 0.5
        return threshold

def evaluate_nn(X_test, y_test):
    model = tf.keras.models.load_model(NN_MODEL_PATH)
    threshold = load_nn_threshold()

    y_fraud_scores = model.predict(x=X_test, verbose=0).flatten()
    y_predicted = (y_fraud_scores >= threshold).astype(int)

    return compute_metrics(y_correct_labels=y_test, y_predicted=y_predicted, y_fraud_scores=y_fraud_scores)    


def main():

    X_test, y_test = load_test_data()

    n_fraud = int(np.sum(y_test == 1))
    n_legit = int(np.sum(y_test == 0))
    fraud_rate = (n_fraud / len(y_test)) * 100

    print(f"\nTest set: {n_legit} LEGITIMATE, {n_fraud} FRAUD")
    print(f"Fraud rate in the test set: {fraud_rate:.5f}%")

    perceptron_evaluation_metrics = evaluate_perceptron(X_test, y_test)
    nn_evaluation_metrics = evaluate_nn(X_test, y_test)

    print_metrics(perceptron_evaluation_metrics, model_name="Perceptron")
    print_metrics(nn_evaluation_metrics, model_name="Artificial Neural Network")
    

if __name__ == "__main__":
    main()
