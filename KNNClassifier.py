from collections import Counter
from Classifier import Classifier
from Distance import Distance
from data import Data

class KNNClassifier(Classifier):
    def __init__(self, k, distance: Distance):
        self.k = k
        self.distance = distance
        self.training_data = []

    def train(self, training_data: list[Data]):
        self.training_data = training_data

    def evaluate(self, test_data: list[Data]):
        predictions = []
        for test_instance in test_data:
            # Calcul des distances entre la donnée de test et toutes les données d'entraînement
            distances = [(train_instance, self.distance.compute(test_instance, train_instance)) for train_instance in self.training_data]
            
            # Tri des distances par ordre croissant
            distances.sort(key=lambda x: x[1])

            # Sélection des k plus proches voisins
            k_nearest_neighbors = distances[:self.k]

            # Vote majoritaire pour déterminer la classe de la donnée de test
            labels = [neighbor[0].label for neighbor in k_nearest_neighbors]
            most_common_label = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common_label)

        return predictions

