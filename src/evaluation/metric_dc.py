from dataclasses import dataclass

@dataclass
class Metric:
    recall: float
    precision: float

    def __init__(self, recall, precision):
        self.recall     = recall
        self.precision  = precision