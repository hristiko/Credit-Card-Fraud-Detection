import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score
from utils.preprocessing import load_data, preprocess_data
from config import NN_MODEL_PATH, NN_THRESHOLD_PATH

def load_saved_threshold():
    with open(NN_THRESHOLD_PATH, "r") as file:
        return float(file.read().strip())

def main():
    os.makedirs("analysis_outputs", exist_ok=True)

    dataset = load_data("data/creditcard.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(dataset)

    model = tf.keras.models.load_model(NN_MODEL_PATH)
    saved_threshold = load_saved_threshold()

    y_fraud_scores = model.predict(x=X_test, verbose=0).flatten()
    y_predicted = (y_fraud_scores >= saved_threshold).astype(int)

    predictions_data = pd.DataFrame({
        "y_correct_labels": y_test,
        "y_fraud_scores": y_fraud_scores,
        "y_predicted": y_predicted
    })

    predictions_data.to_csv(
        "analysis_outputs/nn_test_predictions.csv",
        index=False
    )

    threshold_rows = []

    for threshold in np.arange(0.05, 1.00, 0.01):
        threshold = round(float(threshold), 2)

        y_predicted_threshold = (y_fraud_scores >= threshold).astype(int)

        precision = precision_score(
            y_test,
            y_predicted_threshold,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_predicted_threshold,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_predicted_threshold,
            zero_division=0
        )

        threshold_rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    threshold_metrics_data = pd.DataFrame(threshold_rows)

    threshold_metrics_data.to_csv(
        "analysis_outputs/threshold_metrics.csv",
        index=False
    )

    print("Analysis data saved successfully.")
    print("Saved: analysis_outputs/nn_test_predictions.csv")
    print("Saved: analysis_outputs/threshold_metrics.csv")
    print(f"Saved threshold used by model: {saved_threshold}")


if __name__ == "__main__":
    main()