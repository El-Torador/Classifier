from abc import ABC, abstractmethod
from data import Data

class Classifier(ABC):
    @abstractmethod
    def train(self, dataset: list[Data]):
        pass

    @abstractmethod
    def evaluate(self, data_instance: Data):
        pass