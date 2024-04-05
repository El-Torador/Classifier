import numpy as np
from data import Data

def normalizeZScore(data_vector: list[Data]) -> list[Data]:
    # Calculer la moyenne et l'écart-type pour chaque caractéristique
    mean_values = np.mean([data.features for data in data_vector], axis=0)
    std_values = np.std([data.features for data in data_vector], axis=0)

    # Normaliser chaque caractéristique pour chaque donnée
    for data in data_vector:
        data.features = (data.features - mean_values) / std_values

    return data_vector