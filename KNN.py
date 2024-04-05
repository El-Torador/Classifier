from Classifier import Classifier
from data import Data

class KNN(Classifier):
    def __init__(self, k: int, distance_metric: str = "euclidean"):
        self.k = k
        self.distance_metric = distance_metric
        self.training_data = []

    def train(self, dataset: list[Data]):
        self.training_data = dataset

    def evaluate(self, test_data: list[Data]):
        pass
