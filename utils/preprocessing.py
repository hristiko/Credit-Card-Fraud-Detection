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
    print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    return dataset

def preprocess_data(dataset):

    X = dataset.drop(columns="Class")
    y = dataset["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    scaler = StandardScaler()

    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    columns_to_scale = ["Time", "Amount"]

    X_train_copy[columns_to_scale] = scaler.fit_transform(X_train_copy[columns_to_scale])
    X_test_copy[columns_to_scale] = scaler.transform(X_test_copy[columns_to_scale])

    X_train_final = X_train_copy.values
    X_test_final = X_test_copy.values

    distribution_train_data = get_class_distribution(y_train)
    distribution_test_data = get_class_distribution(y_test)

    fraud_precentage = distribution_train_data.get(1, 0) / sum(distribution_train_data.values()) * 100

    print(f"\nTraining set: {X_train_final.shape[0]} samples")
    print(f"({distribution_train_data.get(0, 0)} legit | {distribution_train_data.get(1,0)} fraud)")


    print(f"\n Test set: {X_test.shape[0]:} samples")
    print(f"({distribution_test_data.get(0, 0):} legit | {distribution_test_data.get(1,0)} fraud)")

    print(f"Fraud ratio in training: {fraud_precentage:.5f}% \n")

    return X_train_final, X_test_final, y_train, y_test

def oversample_minority(X_train, y_train, random_state = RANDOM_SEED):
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

    print(f"Balanced set: {np.sum(y_balanced == 0)} LEGIT, {np.sum(y_balanced == 1)} FRAUD\n")

    return X_balanced, y_balanced

#def get_class_weights(y_train):
#    classes = np.unique(y_train)
#
#    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
#    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
#
#    print(f"Class weights: {weight_dict}")
#    
#    return weight_dict

def get_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
