import json
import torch
from torch import nn
from pathlib import Path
from typing import Optional
from collections import Counter

from brics_toolkit.data_containers import MeasurementData, MeasurementMetadata
from brics_toolkit.data_containers.brv_data_clean import BRVDataClean
from model.model.brics_model import BRICSModel
from brics_toolkit.data_processing import initial_data_processing, extract_features
from brics_toolkit.utils.config import *
from model.data_containers.feature_data import FeatureData

IDENTIFIER_DIR_PATH = Path("./data/identifier/")
IDENTIFIER_DIR_PATH.parent.mkdir(parents=True, exist_ok=True)
IDENTIFIER_FEATURES_PATH = IDENTIFIER_DIR_PATH / "features"
IDENTIFIER_FEATURES_PATH.mkdir(parents=True, exist_ok=True)
IDENTIFIER_RESULTS_PATH = IDENTIFIER_DIR_PATH / "clean"
IDENTIFIER_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

def split_data_into_segments(input_file : Path, BRV_data_clean : BRVDataClean):
    """
        Split the resampled ADC data into segments that contain values from a specific time window,
        and save each segment into a separate JSONL file.

        Parameters
        ----------
        input_file: Path
            The path to the raw input file containing the ADC data.
        BRV_data_clean : BRVDataClean
            The BRVDataClean object containing the cleaned and resampled ADC data and timestamps further
            defined in project's DTP.

        Returns
        -------
        None          
        
        Side Effects
        ------------
        This function creates multiple JSONL files in the "./results" directory, each containing a segment of the ADC data.
    """

    segment_index = 0
    total_segments = int(np.ceil(BRV_data_clean.timestamps[-1] / SEGMENT_LENGTH_MS))
    temp_name = input_file.name
    temp = temp_name.split('.')
    temp.pop()
    filename = ".".join(temp)
    
    for segment_index in range(total_segments):
        segment_start = segment_index * SEGMENT_LENGTH_MS
        segment_end = segment_start + SEGMENT_LENGTH_MS
        with open(f"{IDENTIFIER_RESULTS_PATH}/{filename}_{segment_index}.jsonl", 'w') as o_f:
            for i in range(len(BRV_data_clean.timestamps)):
                if segment_start <= BRV_data_clean.timestamps[i] < segment_end:
                    record = {
                        "timestamp": int(BRV_data_clean.timestamps[i]),
                        "adc_outputs": [BRV_data_clean.adc_data[a][i] for a in range(ADC_COUNT)]
                    }
                    o_f.write(json.dumps(record) + "\n")

def load_inference_features(features_dir: Path) -> torch.Tensor:
    rows = []

    for file_path in sorted(features_dir.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                row = []
                for value in record.values():
                    if isinstance(value, list):
                        row.extend(value)
                    else:
                        row.append(value)
                rows.append(row)

    return torch.tensor(rows, dtype=torch.float32)


class BRICSModelWrapper:
    def __init__(self, model: nn.Module|None, learning_rate: float = 0.001) -> None:
        self.model: Optional[nn.Module] = model
        self.parameters: Optional[dict] = {
            # TODO: those will have to be read from here and handled properly in the future
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
    
    def identify_person(self, input_file: Path):
        if self.model is None:
            raise ValueError("Model is not defined.")
        
        filename = input_file.name

        measurement_data = MeasurementData()
        measurement_metadata = MeasurementMetadata()
        measurement_metadata.filepath_raw = Path(input_file)
        measurement_data.metadata = measurement_metadata    

        initial_data_processing(
            BRV_measurement_data = measurement_data, 
            target_adc = TARGET_ADC, 
            plot_enabled = False
        )
        measurement_metadata.filepath_raw = Path(input_file)
        split_data_into_segments(Path(input_file), measurement_data.data_clean)
        measurement_data.metadata.filepath_clean = Path(f"./data/identifier/clean/{filename}")
        extract_features(
            measurement_data=measurement_data, 
            features_path=str(IDENTIFIER_FEATURES_PATH), 
            results_path=str(IDENTIFIER_RESULTS_PATH), 
            model_flag=True
        )

        features_tensor = load_inference_features(IDENTIFIER_FEATURES_PATH)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(features_tensor)
            predictions = torch.argmax(logits, dim=1)

        most_common_label = Counter(predictions.tolist()).most_common(1)[0][0]
        person = self.get_person(most_common_label)

        # print(self.people_keys)
        print(f"Predicted label: {most_common_label}, corresponding to person: {person}")

        # clean identifier directories after identification
        for file in IDENTIFIER_FEATURES_PATH.iterdir():
            if file.is_file():
                file.unlink()
        for file in IDENTIFIER_RESULTS_PATH.iterdir():
            if file.is_file():
                file.unlink()
        
        return person

    def save_model(self, filepath: Path):
        if self.model is None:
            raise ValueError("Model is not defined.")
        if not self.parameters:
            raise ValueError("Model parameters are not defined.")
        # firstly, save the self.parameters to "parameters.jsonl" + input/output sizes
        payload = {
            "parameters": {
                "optimizer": self.parameters["optimizer"].__class__.__name__,
                "criterion": self.parameters["criterion"].__class__.__name__,
                "learning_rate": self.parameters["learning_rate"]
            },
            "people_keys": self.people_keys,
            "feature_keys": self.feature_keys
        }
        with open(filepath.with_suffix(".jsonl"), "w") as f:
            json.dump(payload, f)
        torch.save(self.model.state_dict(), filepath.with_suffix(".pth"))

    def load_model(self, filepath: Path = Path("../model/model")):
        # load input/output sizes and parameters from a separate file called "parameters.json"
        # (try & except for the model input and output size) 
        # (in case number of features or number of people differs in saved model)
        # TODO: Adam and CrossEntropyLoss are hardcoded for now because they are
        # not JSON serializable, so we will have to handle it later

        try:
            with open(filepath.with_suffix(".jsonl"), "r") as f:
                payload = json.load(f)
                # print(payload)
            self.people_keys = payload["people_keys"]
            self.feature_keys = payload["feature_keys"]
            self.people_keys = {int(k): v for k, v in payload["people_keys"].items()}
            self.feature_keys = {int(k): v for k, v in payload["feature_keys"].items()}
            self.parameters = {}
            self.parameters["learning_rate"] = payload["parameters"]["learning_rate"]
            self.model = BRICSModel(feature_count=len(self.feature_keys), n_classes=len(self.people_keys))
                        
            if self.model is not None:
                state_dict = torch.load(filepath.with_suffix(".pth"))
                self.model.load_state_dict(state_dict)
                self.model.eval()

            if payload["parameters"]["optimizer"] == "Adam":
                self.parameters["optimizer"] = torch.optim.Adam(self.model.parameters(), lr=self.parameters["learning_rate"])
            if payload["parameters"]["criterion"] == "CrossEntropyLoss":
                self.parameters["criterion"] = torch.nn.CrossEntropyLoss()
        except Exception as e:
            print(f"Error loading model or parameters: {e}")
            self.model = None
            self.parameters = {}
            self.people_keys = {}
            self.feature_keys = {}
