from torch import nn
from pathlib import Path

class BRICSModelWrapper:
    def __init__(self) -> None:
        self.model: nn.Module
        self.parameters: dict
        self.people_keys: dict
        self.feature_keys: dict
    
    def get_person(self, label: int) -> str | None:
        if self.people_keys:
            return self.people_keys[label]
        else:
            return None
    
    def identify_person(self, feedData): #TODO during UI integration
        pass

    def save_model(self, filepath: Path): #TODO during model training
        pass

    def load_model(self, filepath: Path = Path("../model/model.pt")):
        pass