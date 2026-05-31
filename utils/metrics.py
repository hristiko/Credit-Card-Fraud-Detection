import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_predicted):

    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)

    confusion_mat = confusion_matrix(y_true, y_predicted, labels=[0,1])
    true_negative, false_positive, false_negative, true_positive = confusion_mat.ravel()

    total_predictions = len(y_true)
    correct_predictions = true_negative + true_positive

    accuracy = correct_predictions / total_predictions
    precision = precision_score(y_true, y_predicted, zero_division=0)
    recall = recall_score(y_true, y_predicted, zero_division=0)
    f1 = f1_score(y_true, y_predicted, zero_division=0)
    false_positive_rate = false_positive / (false_positive + true_negative)


    return {
        "accuracy": round(float(accuracy), 5),
        "precision": round(float(precision), 5),
        "recall": round(float(recall), 5),
        "f1_score": round(float(f1), 5),
        "false_positive_rate": round(float(false_positive_rate), 5),
        "correct_predictions": int(correct_predictions),
        "total_predictions": int(total_predictions),
        "fraud_detected": int(true_positive),
        "fraud_missed": int(false_negative),
        "legit_as_fraud": int(false_positive),
        "true_legitimate": int(true_negative),
        "confusion_matrix": confusion_mat,
    }

def print_metrics(metrics, model_name):

    accuracy_percentage = metrics["accuracy"] * 100
    precision_percentage = metrics["precision"] * 100
    recall_percentage = metrics["recall"] * 100
    f1_percentage = metrics["f1_score"] * 100
    false_positive_rate_percentage = metrics["false_positive_rate"] * 100
    confusion_mat = metrics["confusion_matrix"]

    print(f"\nMODEL: {model_name}")
    print("Overall Performance")
    print(f"Correct predictions: {accuracy_percentage:.2f}%")

    print("\nClassification metrics")
    print(f"Precision: {precision_percentage:.2f}%")
    print(f"Recall: {recall_percentage:.2f}%")
    print(f"F1-score: {f1_percentage:.2f}%")
    print(f"False positive rate: {false_positive_rate_percentage:.2f}%")

    print("\nFraud Detection")
    print(f"Correct detection: {metrics['fraud_detected']}")
    print(f"Missed: {metrics['fraud_missed']}")
    print(f"Legit transactions marked as fraud: {metrics['legit_as_fraud']}")

    print("\nConfusion matrix")
    print(f"  {'':20} Predicted Legit   Predicted Fraud")
    print(f"  {'Actual Legit':<20} {confusion_mat[0][0]:>14,}   {confusion_mat[0][1]:>14,}")
    print(f"  {'Actual Fraud':<20} {confusion_mat[1][0]:>14,}   {confusion_mat[1][1]:>14,}")

    
    
    

        