from typing import Optional
import numpy as np

class FeatureData():
    def __init__(self):
        self.feature_files = []
        self.feature_colors = []
        self.features: Optional[np.ndarray] = None
        self.feature_count = None
        self.features_pca: Optional[np.ndarray] = None

        self.feature_index = 0
        self.feature_keys = {}
        self.person_colors = {} # dictionary for colors of data points for different person labels
        self.person_indices = {} # dictionary holding arrays of indices in feature data for people
        self.person_initials = [] # array holding all initials for labels in legend


def load_feature_data():
    pass

def specific_features_classifier():
    load_feature_data()

def main():
    specific_features_classifier()

if __name__ == "__main__":
    main()