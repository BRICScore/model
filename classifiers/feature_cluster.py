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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Original labels ---
    for person, indices in feature_data.person_indices.items():
        indices = np.asarray(indices, dtype=int)

        axes[0].scatter(
            X[indices, 0],
            X[indices, 1],
            label=str(person),
            alpha=0.7
        )

    axes[0].set_title("Original Labels")
    axes[0].set_xlabel(f"Feature {feature_data.feature_keys[chosen_pair[0]]}")
    axes[0].set_ylabel(f"Feature {feature_data.feature_keys[chosen_pair[1]]}")

    # Sort legend alphabetically
    handles, legend_labels = axes[0].get_legend_handles_labels()
    sorted_items = sorted(zip(legend_labels, handles), key=lambda x: x[0])

    legend_labels_sorted, handles_sorted = zip(*sorted_items)

    axes[0].legend(
        handles_sorted,
        legend_labels_sorted,
        title="Person",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    axes[0].grid(alpha=0.2)

    # --- Right: HDBSCAN clusters ---
    for cluster in np.unique(labels):
        mask = labels == cluster

        axes[1].scatter(
            X[mask, 0],
            X[mask, 1],
            label="Noise" if cluster == -1 else f"Cluster {cluster}",
            marker="x" if cluster == -1 else "o",
            alpha=0.7
        )

    axes[1].set_title("HDBSCAN Clusters")
    axes[1].set_xlabel(f"Feature {feature_data.feature_keys[chosen_pair[0]]}")
    axes[1].set_ylabel(f"Feature {feature_data.feature_keys[chosen_pair[1]]}")
    axes[1].legend(title="Cluster")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

def main():
    hdbscan()

if __name__ == "__main__":
    main()