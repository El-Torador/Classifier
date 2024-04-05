from abc import ABC, abstractmethod
from data import Data

class Distance(ABC):
    @abstractmethod
    def compute(self, data_instance1: Data, data_instance2: Data)-> float:
        pass