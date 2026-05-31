import numpy as np

from utils.preprocessing import load_data, preprocess_data
from utils.metrics import compute_metrics, print_metrics
from models.perceptron import Perceptron
from config import PERCEPTRON_WEIGHTS_PATH, PERCEPTRON_BIAS_PATH

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

    return compute_metrics(y_test, y_predicted)

def main():

    X_test, y_test = load_test_data()

    n_fraud = int(np.sum(y_test == 1))
    n_legit = int(np.sum(y_test == 0))
    fraud_rate = (n_fraud / len(y_test)) * 100

    print(f"\nTest set: {n_legit} LEGITIMATE, {n_fraud} FRAUD")
    print(f"Fraud rate in the test set: {fraud_rate:.5f}%")

    perceptron_evaluation_metrics = evaluate_perceptron(X_test, y_test)

    print_metrics(perceptron_evaluation_metrics, model_name="Perceptron")

if __name__ == "__main__":
    main()
