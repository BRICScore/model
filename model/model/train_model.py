from model.data_containers import *
from .parse_features import parse_features_from_data
from .brics_model import BRICSModel

def train_model():
    model_data = parse_features_from_data()
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