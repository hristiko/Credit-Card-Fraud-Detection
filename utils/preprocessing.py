import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from config import TEST_SIZE, RANDOM_SEED

def load_data(path):
    
    dataset = pd.read_csv(path)
    print(f"Dataset loaded: {dataset.shape[0]:,} rows, {dataset.shape[1]} columns")
    return dataset

def preprocess_data(dataset):

    X = dataset.drop(columns="Class")
    y = dataset["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    columns_to_scale = ["Time", "Amount"]

    X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    X_train = X_train.values
    X_test = X_test.values

    distTrain = get_class_distribution(y_train)
    distTest = get_class_distribution(y_test)

    fraudPrecentage = distTrain.get(1, 0) / sum(distTrain.values()) * 100

    print(
        f"\n Training set: {X_train.shape[0]:, } samples"
        f"({distTrain.get(0, 0):,} legit | {distTrain.get(1,0):,} fraud)"
    )

    print(
        f"\n Test set: {X_test.shape[0]:, } samples"
        f"({distTest.get(0, 0):,} legit | {distTest.get(1,0):,} fraud)"
    )

    print(f"Fraud ratio in training: {fraudPrecentage:.4f}% \n")

    return X_train, X_test, y_train, y_test, scaler
    

def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved in: {path}")

def oversample_minoroty(X_train, y_train, random_state = RANDOM_SEED):
    fraud_indicies = np.where(y_train == 1)[0]
    legit_indicies = np.where(y_train == 0)[0]

    n_fraud = len(fraud_indicies)
    n_legit = len(legit_indicies)

    fraud_repeat_count = n_legit // n_fraud

    print(f"Oversampling: repeating {n_fraud} fraud samples * {fraud_repeat_count}, which is arround {n_fraud * fraud_repeat_count} fraud samples")

    oversampled_fraud_indicies = np.tile(fraud_indicies, fraud_repeat_count)

    all_indicies = np.concatenate([legit_indicies, oversampled_fraud_indicies])

    local_random_generator = np.random.default_rng(random_state)
    local_random_generator.shuffle(all_indicies)

    X_balanced = X_train[all_indicies]
    y_balanced = y_train[all_indicies]

    print(f"Balanced set: {np.sum(y_balanced=0)} legit | np.sum(y_balanced == 1) fraud\n")

    return X_balanced, y_balanced


def get_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
