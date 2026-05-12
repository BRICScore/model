import scipy
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import torch

from classifiers.config import *
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import numpy as np
import json

EPOCHS = 50
LR = 0.001
BATCH_SIZE = 16
LAYERS = 3
SEED = 42
HIDDEN_SIZE = 128
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_adc_data_line(line: str):
    def u2_to_i(b1, b2, b3):
        value = (b1 << 16) | (b2 << 8) | b3
        return value - (1 << 24) if value & (1 << 23) else value

    # bit merging
    def extract_adc_data(start_index):
        return u2_to_i(int(parts[start_index].split(':')[1]),
                       int(parts[start_index + 1].split(':')[1]),
                       int(parts[start_index + 2].split(':')[1]))
    
    parts = line.strip().split(',')
    hour = int(parts[0].split(':')[1])
    minute = int(parts[1].split(':')[1])
    second = int(parts[2].split(':')[1])
    millisecond = int(parts[3].split(':')[1])
    ms_timestamp = (hour * 3600 + minute * 60 + second) * 1000 + millisecond

    adc_outputs = [extract_adc_data(4 + i * 3) for i in range(ADC_COUNT)]   # 4 - skip timestamp, i*3 - each ADC has 3 bytes of data 
    return ms_timestamp, adc_outputs

def handle_input_data(input_file : Path):
    def adc_to_voltage(adc_value):
        return (adc_value + 2**23) * 10**(-9) * 23.84
    
    first_timestamp = None
    timestamps = np.array([])
    adc_output_data = [np.array([]) for _ in range(ADC_COUNT)]
    with open(f"{input_file}", 'r') as i_f:
        for line in i_f:
            ms_timestamp, adc_outputs = parse_adc_data_line(line)
            if first_timestamp is None:
                first_timestamp = ms_timestamp
            timestamps = np.append(timestamps, ms_timestamp - first_timestamp)
            for i, v in enumerate(adc_outputs):
                adc_output_data[i] = np.append(adc_output_data[i], round(adc_to_voltage(v), 10))
    return timestamps, adc_output_data

def process_raw_file(input_file: Path):
    timestamps, adc_normalized_data = handle_input_data(input_file)

    for i in range(ADC_COUNT):
            mean_voltage = np.mean(adc_normalized_data[i])
            adc_normalized_data[i] -= mean_voltage

    np_adc_normalized_data = np.vstack(
        adc_normalized_data
    )

    return timestamps, np_adc_normalized_data

def split_data_into_segments(input_file : Path, adc_data: np.ndarray, timestamps: np.ndarray):
    segment_index = 0
    total_segments = int(np.ceil(timestamps[-1] / SEGMENT_LENGTH_MS))
    filename = str(input_file).split("_")
    time = filename[1]
    person = filename[2]
    condition = filename[3]
    no_of_sample = filename[4]

    for segment_index in range(total_segments):
        seg_data = []
        seg_timestamps = []
        segment_start = segment_index * SEGMENT_LENGTH_MS
        segment_end = segment_start + SEGMENT_LENGTH_MS
        # TODO: optimize
        for i in range(len(timestamps)):
            if segment_start <= timestamps[i] < segment_end:
                seg_data.append([adc_data[a][i] for a in range(ADC_COUNT)])
                seg_timestamps.append(timestamps[i] - segment_start)
                
        resampled_data, resampled_timestamps = resample_adc_data_and_timestamps(np.array(seg_data).T, np.array(seg_timestamps), RESAMPLED_NODE_COUNT)
        print(resampled_timestamps[0])

        with open(f"./classifiers/classifiers_data/clean_{time}_{segment_index}_{person}_{condition}_{no_of_sample.split('.')[0]}.jsonl", 'w') as o_f:
            for i in range(len(resampled_timestamps)):
                record = {
                        "timestamp": resampled_timestamps[i],
                        "adc_outputs": [resampled_data[a][i] for a in range(ADC_COUNT)]
                    }
                o_f.write(json.dumps(record) + "\n")

def resample_data(y, new_len):
    x_old = np.linspace(0, 1, len(y))
    x_new =  np.linspace(0, 1, new_len)
    f = interp1d(x_old, y, kind='cubic')
    return f(x_new)

def resample_adc_data_and_timestamps(data, timestamps, resampled_node_count):
    signal_duration = timestamps[-1] - timestamps[0]
    print(f"Signal duration: {signal_duration} ms, resampled_node_count: {resampled_node_count}")
    resampled_timestamps = resample_data(timestamps, resampled_node_count)
    resampled_data = np.vstack([
        resample_data(data[i], resampled_node_count)
        for i in range(ADC_COUNT)
    ])
    return resampled_data, resampled_timestamps

def get_person_form_filename(filename: str):
    return filename.split("_")[3]

def load_segment_file(segment_file: Path):
    timestamps = []
    adc_outputs = []
    with open(segment_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            timestamps.append(record["timestamp"])
            adc_outputs.append(record["adc_outputs"])
    return np.array(timestamps), np.array(adc_outputs)

class SegmentDataset(Dataset):
    def __init__(self, samples, labels):
        self.x = torch.tensor(np.array(samples, dtype=np.float32))
        self.y = torch.tensor(np.array(labels, dtype=np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, layer_dim=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1)
        out = self.head(out)
        return out

def labeled_samples(data_segments_dir: Path):
    files = list(data_segments_dir.glob("*.jsonl"))
    samples, labels, groups = [], [], []
    
    unique_labels = sorted(set([get_person_form_filename(f.name) for f in files]))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    print(unique_labels)

    for file in files:
        source_file_id = "_".join(file.name.split("_")[:2] + file.name.split("_")[3:5] + [file.name.split("_")[5].split(".")[0]])
        # source_file_id = "_".join(file.name.split("_")[:2]
        _, adc_outputs = load_segment_file(file)
        samples.append(adc_outputs.astype(np.float32))
        labels.append(label_to_id[get_person_form_filename(file.name)])
        groups.append(source_file_id)
        # print(groups)

    return samples, labels, groups, label_to_id, id_to_label

def create_dataloaders(samples, labels, groups, batch_size=BATCH_SIZE, seed=SEED):
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        samples, 
        labels, 
        test_size=0.20, 
        stratify=labels, 
        random_state=seed
    )

    all_train_data = np.concatenate(x_train_raw, axis=0)
    mean = np.mean(all_train_data, axis=0)
    std = np.std(all_train_data, axis=0) + 1e-9
    
    def norm(data_list):
        return [(s - mean) / std for s in data_list]

    train_ds = SegmentDataset(norm(x_train_raw), y_train)
    test_ds = SegmentDataset(norm(x_test_raw), y_test)

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    from collections import Counter
    print(f"Test Class Distribution: {Counter(y_test)}")

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False))

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(x_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        scheduler.step(accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy:.4f}")

def lstm_classification(data_segments_dir: Path):
    samples, labels, groups, label_to_id, id_to_label = labeled_samples(data_segments_dir)
    train_loader, test_loader = create_dataloaders(samples, labels, groups, batch_size=8)

    model = LSTMModel(
        input_dim=ADC_COUNT,
        hidden_dim=HIDDEN_SIZE,
        layer_dim=LAYERS,
        output_dim=len(label_to_id),
        dropout=DROPOUT
    )

    model.to(DEVICE)
    print(DEVICE)
    train_model(model, train_loader, test_loader, epochs=EPOCHS, lr=LR)
    return model, test_loader, label_to_id, id_to_label

def results_and_plot(model, test_loader, label_to_id, id_to_label):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    all_labels_ids = sorted(list(id_to_label.keys()))
    class_names = [id_to_label[i] for i in all_labels_ids]
    cm = confusion_matrix(y_true, y_pred, labels=all_labels_ids)

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test Accuracy: {sum(np.array(y_true) == np.array(y_pred)) / len(y_true):.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, 
        y_pred, 
        labels=all_labels_ids,
        target_names=class_names, 
        digits=4, 
        zero_division=0))

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()