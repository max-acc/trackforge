"""
Module for implementing a Relative Deviation based Converger.
"""

from src.training.converger import Converger


class RelativeLossConverger(Converger):
    """
    Model converges, if the relative change of the loss over the previous two last epochs is small.
    """
    def __init__(self, min_epoch):
        super().__init__(min_epoch)
        self.epsilon = 0.01 # TODO move to config

    def has_converged(self, epoch) -> bool:
        return (super().has_converged(epoch) and
                epoch >= 2 and
                abs(self.train_losses[-2] - self.train_losses[-1]) / self.train_losses[-1] < self.epsilon)
