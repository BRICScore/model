from flask import json
import torch
from torch import nn
from pathlib import Path
from typing import Optional

class BRICSModelWrapper:
    def __init__(self, model: nn.Module|None, learning_rate: float = 0.001) -> None:
        self.model: Optional[nn.Module] = model
        self.parameters: Optional[dict] = {
            "optimizer": "Adam",
            "criterion": "CrossEntropyLoss",
            "learning_rate": learning_rate
        }
        self.people_keys = {}
        self.feature_keys = {}

    def get_person(self, label: int) -> Optional[str]:
        if self.people_keys:
            return self.people_keys.get(label, None)
        else:
            return None
    
    def identify_person(self, feedData):
        pass

    def save_model(self, filepath: Path):
        if self.model is None:
            raise ValueError("Model is not defined.")
        # firstly, save the self.parameters to "parameters.jsonl" + input/output sizes
        payload = {
            "parameters": self.parameters,
            "people_keys": self.people_keys,
            "feature_keys": self.feature_keys
        }
        with open(filepath.with_suffix(".jsonl"), "w") as f:
            json.dump(payload, f)
        torch.save(self.model.state_dict(), filepath.with_suffix(".pth"))

    def load_model(self, filepath: Path = Path("../model/model.pth")):
        # load input/output sizes and parameters from a separate file called "parameters.json"
        # (try & except for the model input and output size) 
        # (in case number of features or number of people differs in saved model)
        # TODO: Adam and CrossEntropyLoss are hardcoded for now because they are
        # not JSON serializable, so we will have to handle it later

        try:
            with open(filepath.with_suffix(".jsonl"), "r") as f:
                payload = json.load(f)
            self.parameters = payload["parameters"]
            self.people_keys = payload["people_keys"]
            self.feature_keys = payload["feature_keys"]
            
            if self.model is not None:
                state_dict = torch.load(filepath.with_suffix(".pth"))
                self.model.load_state_dict(state_dict)
                self.model.eval()
        except Exception as e:
            print(f"Error loading model or parameters: {e}")
            self.model = None
            self.parameters = {}
            self.people_keys = {}
            self.feature_keys = {}
