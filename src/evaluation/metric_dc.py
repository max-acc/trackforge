from dataclasses import dataclass

@dataclass
class Metric:
    accuracy: float
    recall: float
    precision: float

class MetricList(list[Metric]):
    def __init__(self, metrics: list[Metric] | None = None):
        super().__init__(metrics or [])

    def append(self, metric: Metric):
        if not isinstance(metric, Metric):
            raise TypeError("Expected Metric, got {}".format(type(metric)))
        super().append(metric)

    @property
    def get_accuracy(self) -> list[float]:
        return [metric.accuracy for metric in self]

    @property
    def get_recall(self) -> list[float]:
        return [metric.recall for metric in self]

    @property
    def get_precision(self) -> list[float]:
        return [metric.precision for metric in self]