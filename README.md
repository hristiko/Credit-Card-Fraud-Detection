# Credit Card Fraud Detection

This is an ML project for studting purposes, which detects fraudulent credit card transactions. The dataset is highly imbalanced, meaning that legitimate transactions are much more common than fraud transactions.

Link to the dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The project compares two models:

* Perceptron
* Artificial Neural Network

## Required Libraries

Install the required libraries before running the scripts:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

## Libraries Used

```text
numpy        - numerical operations
pandas       - loading and handling the dataset
scikit-learn - preprocessing, metrics, Perceptron
tensorflow   - Artificial Neural Network
matplotlib   - plotting graphs
```

## Running the Project

Before running the evaluation script, both models must be trained first. This is necessary because the evaluation script uses the saved trained models.

Run the training scripts first:

```bash
python -m training.train_perceptron
python -m training.train_neural_network
```
Then run the evaluation script:

```
python -m evaluation
```

## Notes

The `Class` column is the target:

```text
0 = legitimate transaction
1 = fraudulent transaction
```

`Time` and `Amount` are scaled using `StandardScaler`.

The Perceptron uses random oversampling of fraud cases because the dataset is highly imbalanced.

The Artificial Neural Network is the main model and uses techniques such as focal loss, callbacks, and threshold tuning to improve fraud detection.
