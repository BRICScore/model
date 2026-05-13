from feature_data import FeatureData
from brics_model_wrapper import BRICSModelWrapper

class ModelData:
    def __init__(self, feature_data: FeatureData) -> None:
        self.feature_data = feature_data
        self.model_wrapper: BRICSModelWrapper
        self.__setup_loaders()

    def __setup_loaders(self): #TODO after parser scripts
        pass

    def scale_data(self): #TODO after parser scripts
        pass