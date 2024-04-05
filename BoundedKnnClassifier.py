from collections import Counter
from KNN import KNN
from Distance import Distance
from data import Data

class BoundedKNNClassifier(KNN):
  def __init__(self, k: int, bound: int, distance: Distance):
    self.k = k
    self.distance = distance
    self.bound = bound
    self.training_data = []

  def train(self, training_data: list[Data]):
    self.training_data = training_data

  def evaluate(self, test_data: list[Data]):
    predictions = []

    for test_instance in test_data:
      # Calcul des distances entre la donnée de test et toutes les données d'entraînement
      distances = self.calc_distance(test_instance)

      # Tri des distances par ordre croissant celles qui sont inférieures ou égales à la borne
      distances.sort(key=lambda x: x[1] if x[1] <= self.bound else float('inf'))

      # S'il n'y a pas suffisament de voisins dans la borne de distance, on augmente la borne
      while len(distances) < self.k:
        self.bound *= 1.1
        distances.sort(key=lambda x: x[1] if x[1] <= self.bound else float('inf'))
      # Selection des k plus proches voisins
      k_nearest_neighbors = sorted(distances[:self.k], key=lambda x: x[1])[:self.k]

      # Vote majoritaire pour déterminer la classe de la donnée de test
      labels = [neighbor[0].label for neighbor in k_nearest_neighbors]
      most_common_label = Counter(labels).most_common(1)[0][0]

      predictions.append(most_common_label)

    return predictions
