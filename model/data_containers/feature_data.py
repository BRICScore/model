import numpy as np
from torch import Tensor, tensor

# this is a basic struct therefore it can be represented with
# a data class of some sort
class FeatureData:
    def __init__(self, data: Tensor = tensor(np.array([])), labels: Tensor = tensor(np.array([]))) -> None:
        self.data = data
        self.labels = labels
        