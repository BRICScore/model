from pathlib import Path

from sklearn.metrics import confusion_matrix

from model.data_containers import *
from .parse_features import parse_features_from_data
from .brics_model import BRICSModel
from .hyperparameters import *

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader

def display_confusion_matrix(classes, preds):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    cm = confusion_matrix(classes, preds)
    # print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.show()

    correct = (classes == preds)

    plt.figure(figsize=(6,5))

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data = parse_features_from_data(Path("./data/features"))
    # print(model_data.feature_data.data)
    # print(model_data.feature_data.data.size())
    # print(model_data.feature_data.labels)
    # print(model_data.feature_data.labels.size())

    # print(model_data.model_wrapper.people_keys)
    class_count = len(model_data.model_wrapper.people_keys)
    # print(model_data.model_wrapper.feature_keys)
    feature_count = len(model_data.model_wrapper.feature_keys)
    model_data.model_wrapper.model = BRICSModel(feature_count=feature_count, n_classes=class_count)

    # start the training here
    # 1. Setup DataLoaders in model data
    # 2. Choose optimizer and criterion and put it into model wrapper
    # 3. Rest is in the training issue
    model = model_data.model_wrapper.model
    model.to(device)

    # ---- loss + optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if model_data.model_wrapper.parameters is None:
        model_data.model_wrapper.parameters = {}
    model_data.model_wrapper.parameters["criterion"] = criterion
    model_data.model_wrapper.parameters["optimizer"] = optimizer

    # data loaders
    dataset_size = len(model_data.feature_data.data)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    data_indices = torch.randperm(dataset_size)
    train_data_indices = data_indices[:train_size]
    test_data_indices = data_indices[train_size:]

    train_dataset = TensorDataset(model_data.feature_data.data[train_data_indices], model_data.feature_data.labels[train_data_indices])
    model_data.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = TensorDataset(model_data.feature_data.data[test_data_indices], model_data.feature_data.labels[test_data_indices])
    model_data.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ---- training loop ----
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        for (features, labels) in model_data.train_loader:

            features.to(device)
            labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()
    
        # if epoch % 20 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item():.4f}", end=" ")
        #     preds = torch.argmax(outputs, dim=1)
        #     acc = (preds == labels).float().mean()
        #     print(f"Model accuracy is: {acc.item()}")

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = np.zeros(class_count)
        n_class_samples = np.zeros(class_count)
        print()
        print("-----Beginning of test section-----")
        # for (features, labels) in model_data.test_loader:
        #     features.to(device)
        #     labels.to(device)

        #     logits = model(features)
        #     preds = torch.argmax(logits, dim=1)
        #     acc = (preds == labels).float().mean()
        #     print(f"Model accuracy is: {acc.item()}")
        features, labels = model_data.feature_data.data[test_data_indices], model_data.feature_data.labels[test_data_indices]
        logits = model(features)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        print(f"Model accuracy is: {acc.item()}")
        if acc.item() > 0.85:
            display_confusion_matrix(labels, preds);
    return model_data