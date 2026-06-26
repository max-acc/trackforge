from dataclasses import dataclass

@dataclass
class Metric:
    accuracy: float
    recall: float
    precision: float

    def __init__(self, accuracy, recall, precision):
        self.accuracy = accuracy
        self.recall     = recall
        self.precision  = precision