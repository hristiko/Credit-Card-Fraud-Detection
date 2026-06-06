import pandas as pd
import matplotlib.pyplot as plt


def main():
    threshold_metrics_data = pd.read_csv(
        "analysis_outputs/threshold_metrics.csv"
    )

    best_f1_row = threshold_metrics_data.loc[
        threshold_metrics_data["f1_score"].idxmax()
    ]

    best_threshold = best_f1_row["threshold"]
    best_f1 = best_f1_row["f1_score"]

    plt.figure(figsize=(10, 6))

    plt.plot(
        threshold_metrics_data["threshold"],
        threshold_metrics_data["precision"],
        label="Precision"
    )

    plt.plot(
        threshold_metrics_data["threshold"],
        threshold_metrics_data["recall"],
        label="Recall"
    )

    plt.plot(
        threshold_metrics_data["threshold"],
        threshold_metrics_data["f1_score"],
        label="F1-score"
    )

    plt.axvline(
        x=best_threshold,
        linestyle="--",
        label=f"Best F1 threshold = {best_threshold:.2f}"
    )

    plt.title("Threshold vs Precision, Recall and F1-score")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "analysis_outputs/threshold_vs_metrics.png",
        dpi=300
    )

    plt.show()

    print(f"Best threshold by F1-score: {best_threshold:.2f}")
    print(f"Best F1-score: {best_f1:.4f}")


if __name__ == "__main__":
    main()