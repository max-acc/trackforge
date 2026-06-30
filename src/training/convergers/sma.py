"""
Module for implementing a Simple Moving Average based Converger.
"""

from src.training.converger import Converger

class SimpleMovingAverageConverger(Converger):
    """
    Simple Moving Average Converger.

    Computes the simple moving average over a given window size, and converges if current loss is smaller than a given
    epsilon.
    """
    def __init__(self, min_epoch):
        super().__init__(min_epoch)
        self.epsilon = 0.01
        self.window_size = 5

    def has_converged(self, epoch) -> bool:
        return (
            super().has_converged(epoch) and
            epoch >= 2 and
            epoch >= self.window_size and
            abs(self.sma() - self.train_losses[-1]) < self.epsilon
        )

    def sma(self) -> float:
        """
        Computes the simple moving average over a given window size.

        :return:    The SMA value.
        """
        win_length = min(len(self.train_losses), self.window_size)
        cumsum = 0
        for i in range(win_length):
            cumsum += self.train_losses[-(i+1)]

        return cumsum / win_length
