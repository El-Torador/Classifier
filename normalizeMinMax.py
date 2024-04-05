import numpy as np
from data import Data

def normalizeMinMax(data_vector: list[Data]) -> list[Data]:
    # Récupérer les valeurs minimales et maximales pour chaque caractéristique
    min_values = np.min([data.features for data in data_vector], axis=0)
    max_values = np.max([data.features for data in data_vector], axis=0)

    # Normaliser chaque caractéristique pour chaque donnée
    for data in data_vector:
        data.features = (data.features - min_values) / (max_values - min_values)
    
    return data_vector