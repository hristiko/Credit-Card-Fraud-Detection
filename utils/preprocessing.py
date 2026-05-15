import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from config import TEST_SIZE, RANDOM_SEED

def loadData(path):
    
    dataset = pd.read_csv(path)
    print(f"Dataset loaded: {dataset.shape[0]:,} rows, {dataset.shape[1]} columns")
    return dataset

def preprocessData(dataset):

    X = dataset.drop(columns="Class")
    y = dataset["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    scalar = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    columns_to_scale = ["Time", "Amount"]

    X_train[columns_to_scale] = scalar.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scalar.transform(X_test[columns_to_scale])

    X_train = X_train.values
    X_test = X_test.values

    distTrain = get_class_distribution(y_train)
    distTest = get_class_distribution(y_test)

    fraudPrecentage = distTrain.get(1, 0) / sum(distTrain.values()) * 100

    print(
        f"\n Training set: {X_train.shpae[0]:, } samples"
        f"({distTrain.get(0, 0):,} legit | {distTrain.get(1,0):,} fraud)"
    )

    print(
        f"\n Test set: {X_test.shpae[0]:, } samples"
        f"({distTest.get(0, 0):,} legit | {distTest.get(1,0):,} fraud)"
    )

    print(f"Fraud ratio in training: {fraudPrecentage:.4f}% \n")

    return X_train, X_test, y_train, y_test, scalar

def save_scalar(scalar, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(scalar, f)

    print(f"Scalar saved in: {path}")