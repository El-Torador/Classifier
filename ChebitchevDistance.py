import numpy as np
from Distance import Distance
from data import Data

class ChebyshevDistance(Distance):
    def compute(self, data1: Data, data2: Data):
        features1 = np.array(data1.features)
        features2 = np.array(data2.features)
        distance = np.max(np.abs(features1 - features2))
        return distance