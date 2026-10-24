import itertools
from typing import Optional
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

import torch
from torch import nn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from specific_features_classifier import feature_loading, establish_best_features, choose_class_features, FeatureData

import json
import os
import sklearn
FEATURES_PATH = './data/features'
CLEAN_PATH = '/data/clean'
RAW_PATH = '/data/raw'
EPOCHS = 10_000

def hdbscan():
    print(sklearn.__version__)
    feature_data = FeatureData()

    feature_data.features = feature_loading(feature_data)
    cov_matrix = np.cov(feature_data.features_scaled, rowvar=False)

    best_pair, best_value = establish_best_features(feature_data=feature_data, cov=cov_matrix)
    chosen_pair = choose_class_features(feature_data=feature_data, best_pair=best_pair)

    X = feature_data.features[:, [chosen_pair[0], chosen_pair[1]]]

    X_scaled = StandardScaler().fit_transform(X)

    # Perform HDBSCAN clustering
    model = HDBSCAN(
        min_cluster_size=10,
        min_samples=5
    )

    labels = model.fit_predict(X_scaled)

    # Count clusters and noise points
    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(labels == -1)

    print(f"Number of clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")
    print(f"Total points: {len(labels)}")

    # Plot clusters
    plt.figure(figsize=(10, 7))

    for cluster in np.unique(labels):
        mask = labels == cluster

        if cluster == -1:
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                c="lightgray",
                marker="x",
                label="Noise",
                alpha=0.7
            )
        else:
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                label=f"Cluster {cluster}",
                alpha=0.7
            )

    plt.xlabel(feature_data.feature_keys[chosen_pair[0]])
    plt.ylabel(feature_data.feature_keys[chosen_pair[1]])
    plt.title("HDBSCAN Clustering")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()

def main():
    hdbscan()

if __name__ == "__main__":
    main()