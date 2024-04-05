from Classifier import Classifier
from Distance import Distance
from data import Data

class KNN(Classifier):
    def __init__(self, k: int, distance: Distance):
        self.k = k
        self.distance = distance
        self.training_data = []
    
    def calc_distance(self, test_instance: Data) -> list[tuple[Data, float]]:
        distance = [(train_instance, self.distance.compute(test_instance, train_instance)) for train_instance in self.training_data]
        return distance


    def train(self, dataset: list[Data]):
        self.training_data = dataset

    def evaluate(self, test_data: list[Data]):
        pass
