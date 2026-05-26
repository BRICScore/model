import json
from model.data_containers import *
import os
from pathlib import Path
from torch import Tensor, tensor
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def scale_data(data: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    joblib.dump(scaler, './model/scaler.pkl')
    return scaled

def create_indices_for_features(record: str) -> dict:
    feature_keys = {}

    i = 0
    json_line = json.loads(record)
    for key in json_line:
        if isinstance(json_line[key], list):
            index = 1
            for val in json_line[key]:
                feature_keys[i] = key + str(index)
                index += 1
                i += 1
        else:
            feature_keys[i] = key
            i += 1
    return feature_keys

def parse_record(record):
    features = json.loads(record)
    feature_vector = []
    for key, val in features.items():
        if type(val) == list:
            for number in val:
                feature_vector.append(number)
        else:
            feature_vector.append(val)
    return feature_vector

def parse_features_from_data(input_file: Path) -> ModelData:
    model_wrapper = BRICSModelWrapper(None)
    people_keys = {}
    feature_keys = {}
    first_go = True
    row_index = 0
    person_index = 0
    data = []
    labels = []
    with os.scandir(input_file) as files:
        for file in files:
            if first_go:
                with open(file, "r") as f:
                    f.readline()
                    feature_keys = create_indices_for_features(f.readline())
            with open(file, "r") as f:
                counter = 0
                metadata = json.loads(f.readline())
                record = f.readline()
                while record:
                    labels.append(person_index)
                    feature_vector = parse_record(record)
                    data.append(feature_vector)
                    record = f.readline()
            people_keys[person_index] = metadata["person_id"]
            person_index += 1
    model_wrapper.feature_keys = feature_keys
    model_wrapper.people_keys = people_keys
    np_data = np.array(scale_data(np.array(data)))
    np_labels = np.array(labels)
    labele = tensor(np_labels, dtype=torch.long)
    print(labele.shape)
    feature_data = FeatureData(data=tensor(np_data, dtype=torch.float32), labels=labele)
    # print(feature_data.data)
    # print(feature_data.labels)
    model_data = ModelData(feature_data=feature_data, model_wrapper=model_wrapper)
    return model_data

