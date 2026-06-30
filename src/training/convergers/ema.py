"""
Module for implementing a Exponential Moving Average based Converger.
"""

from src.training.converger import Converger


class ExponentialMovingAverageConverger(Converger):
    """
    Exponential Moving Average Converger.

    Computes the exponential moving average over the whole train cycle, and converges when the loss is smaller than the
    specified epsilon.
    """
    def __init__(self, min_epoch):
        super().__init__(min_epoch)
        self.alpha = 0.5
        self.s = 0.0
        self.epsilon = 0.01

    def has_converged(self, epoch) -> bool:
        return (
            super().has_converged(epoch) and
            epoch >= 2 and
            abs(self.ema(self.train_losses[-1]) - self.train_losses[-1]) < self.epsilon
        )

    def ema(self, loss) -> float:
        """
        Computes the actual EMA value.

        :param loss:    The loss of the current epoch.
        :return:    The EMA value.
        """
        self.s = loss \
            if len(self.train_losses) <= 1 \
            else self.alpha * loss + (1 - self.alpha) * self.s

        return self.s
