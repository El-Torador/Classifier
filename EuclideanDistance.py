import numpy as np
from Distance import Distance
from data import Data

class EuclideanDistance(Distance):
    def compute(self, data1: Data, data2: Data):
        # Conversion des features en array numpy
        features1 = np.array(data1.features)
        features2 = np.array(data2.features)

        # Calcul de la distance euclidienne
        distance = np.linalg.norm(features1 - features2)

        return distance
