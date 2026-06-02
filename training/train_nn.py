import os
import time
import numpy as np
from sklearn.metrics import f1_score
from utils.preprocessing import load_data, preprocess_data, save_scaler, get_class_weights
from utils.metrics import compute_metrics, print_metrics
from models.neural_network import build_model
from config import NN_EPOCHS, NN_BATCH_SIZE, SCALER_PATH, NN_MODEL_PATH, NN_THRESHOLD_PATH

def find_best_threshold(y_correct_labels, y_fraud_scores):
    best_threshold = None
    best_f1 = 0.0

    for threshold in np.arange(0.05, 1, 0.01):
        y_predicted = (y_fraud_scores >= threshold).astype(int)
        f1 = f1_score(y_correct_labels, y_predicted, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(threshold), 2)

    print(f"\nBest threshold: {best_threshold:3f}")
    print(f"F1 = {best_f1:4f} on training data")
    return best_threshold

def main():
    print("\n--------------------------------")
    print("Neural network trianing")

    dataset = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset)

    save_scaler(scaler, SCALER_PATH)

    class_weights = get_class_weights(y_train)

    model = build_model(X_train.shape[1])
    model.summary()
    start_time = time.time()
    model.fit(x=X_train, y=y_train, epochs=NN_EPOCHS, batch_size=NN_BATCH_SIZE, validation_split=0.1, class_weight = class_weights, verbose=1)
    training_time = time.time() - start_time

    train_fraud_scores = model.predict(x=X_train, verbose=0).flatten()
    best_threshold = find_best_threshold(y_correct_labels=y_train, y_fraud_scores=train_fraud_scores)

    os.makedirs(os.path.dirname(NN_THRESHOLD_PATH), exist_ok=True)
    with open(NN_THRESHOLD_PATH, "w") as f:
        f.write(str(best_threshold))

    y_fraud_scores = model.predict(x=X_test, verbose=0).flatten()
    y_predicted = (y_fraud_scores >= best_threshold).astype(int)

    metrics = compute_metrics(y_correct_labels=y_test, y_predicted=y_predicted, y_fraud_scores=y_fraud_scores, training_time=training_time)

    print_metrics(metrics, model_name="Artificial Neural Network")

    os.makedirs(os.path.dirname(NN_MODEL_PATH), exist_ok=True)
    model.save(NN_MODEL_PATH)

if __name__ == "__main__":
    main()