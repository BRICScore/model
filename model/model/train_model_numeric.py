# Training for Identification

from pathlib import Path

from sklearn.metrics import confusion_matrix

from model.data_containers import *
from .parse_features import parse_features_from_data
from .brics_model import BRICSModel
from .hyperparameters import *

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor, tensor
from torch.utils.data import TensorDataset, DataLoader

def display_confusion_matrix(classes, preds, accuracy, show=True):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    cm = confusion_matrix(classes, preds)
    # print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    plt.title(f"Confusion Matrix, accuracy: {accuracy*100:.2f}%")
    if show:
        plt.show()
    else:
        plt.savefig("confusion_matrix.jpg")
        plt.close()

    correct = (classes == preds)

    plt.figure(figsize=(6,5))

def display_numeric_error(values, preds, mse, show=True):
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # Convert tensors to 1D NumPy arrays
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    values = np.asarray(values).flatten()
    preds = np.asarray(preds).flatten()

    fig, ax = plt.subplots(figsize=(6, 3))

    # Actual vs. predicted values
    ax.scatter(values, preds, alpha=0.7, color="blue",
               edgecolors="white", linewidths=0.5)

    # Ideal prediction line
    min_val = min(values.min(), preds.min())
    max_val = max(values.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title(f"Numeric Prediction, average error: {float(mse):.4f}")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Equal scaling makes deviations from the diagonal easier to interpret
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if show:
        plt.show()
    else:
        fig.savefig("confusion_matrix.jpg", dpi=150, bbox_inches="tight")

    plt.close(fig)

def train_model(training_type="Identification"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data = parse_features_from_data(Path("./data/features"), training_type=training_type)
    print(model_data.feature_data.labels)
    # print(model_data.feature_data.data)
    # print(model_data.feature_data.data.size())
    # print(model_data.feature_data.labels)
    # print(model_data.feature_data.labels.size())

    # print(model_data.model_wrapper.people_keys)
    # class_count = len(model_data.model_wrapper.people_keys) if training_type == "Identification" else 2
    # print(model_data.model_wrapper.feature_keys)
    feature_count = len(model_data.model_wrapper.feature_keys)
    model_data.model_wrapper.model = BRICSModel(feature_count=feature_count, n_classes=1) # 1 because of a single numeric output

    # start the training here
    # 1. Setup DataLoaders in model data
    # 2. Choose optimizer and criterion and put it into model wrapper
    # 3. Rest is in the training issue
    model = model_data.model_wrapper.model
    model.to(device)

    # ---- loss + optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

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

            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()
    
        # if epoch % 20 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss.item():.4f}", end=" ")
        #     preds = torch.argmax(outputs, dim=1)
        #     acc = (preds == labels).float().mean()
        #     print(f"Model accuracy is: {acc.item()}")

    with torch.no_grad():
        print()
        print("-----Beginning of test section-----")

        features, labels = model_data.feature_data.data[test_data_indices], model_data.feature_data.labels[test_data_indices]
        preds = model(features).squeeze(-1)
        avg_error = (preds - labels).float().mean()
        mse = nn.functional.mse_loss(preds.float(), labels.float())
        print(f"MSE is: {mse.item()}")
        # if acc.item() > 0.85:
        display_numeric_error(labels, preds, mse=mse, show=False)
    return model_data