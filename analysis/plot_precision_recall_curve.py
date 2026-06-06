import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score


def main():
    predictions_data = pd.read_csv(
        "analysis_outputs/nn_test_predictions.csv"
    )

    y_correct_labels = predictions_data["y_correct_labels"]
    y_fraud_scores = predictions_data["y_fraud_scores"]

    precision, recall, thresholds = precision_recall_curve(
        y_correct_labels,
        y_fraud_scores
    )

    pr_auc = average_precision_score(
        y_correct_labels,
        y_fraud_scores
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.5f}"
    )

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "analysis_outputs/precision_recall_curve.png",
        dpi=300
    )

    plt.show()

    print(f"PR-AUC: {pr_auc:.5f}")


if __name__ == "__main__":
    main()