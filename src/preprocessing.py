import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(train_path, test_path, column_names):
    # Load and prepare the KDD Cup dataset.
    train_data = pd.read_csv(train_path, names=column_names)
    test_data = pd.read_csv(test_path, names=column_names)
    return train_data, test_data

def select_features(data, numeric_columns):
    # Select relevant numeric features from the dataset.
    selected_columns = numeric_columns + ['label']
    return data[selected_columns]

def preprocess_data(X_train, X_test, n_samples=200, n_components=4):
    # Preprocess data with scaling and PCA dimensionality reduction.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train_scaled[:n_samples])
    X_test_reduced = pca.transform(X_test_scaled)

    return X_train_reduced, X_test_reduced
