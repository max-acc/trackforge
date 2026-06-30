"""
Modules for handling evaluation metrics.
"""

from dataclasses import dataclass


@dataclass
class Metric:
    """
    Dataclass resembling various metrics for evaluation.
    """

    accuracy: float
    recall: float
    precision: float


class MetricList(list[Metric]):
    """
    List type for handling lists of type `Metric`.
    """

    def __init__(self, metrics: list[Metric] | None = None):
        super().__init__(metrics or [])

    def append(self, metric: Metric):
        if not isinstance(metric, Metric):
            raise TypeError(f"Expected Metric, got {type(metric)}")
        super().append(metric)

    @property
    def get_accuracy(self) -> list[float]:
        """
        Extract all accuracies from the metrics.

        :return:    A list containing the accuracies.
        """
        return [metric.accuracy for metric in self]

    @property
    def get_recall(self) -> list[float]:
        """
        Extract all recalls from the metrics.

        :return:    A list recall the accuracies.
        """
        return [metric.recall for metric in self]

    @property
    def get_precision(self) -> list[float]:
        """
        Extract all precisions from the metrics.

        :return:    A list precision the accuracies.
        """
        return [metric.precision for metric in self]
