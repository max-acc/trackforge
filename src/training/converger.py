from typing import List

class Converger():
    """
    Creates a converger for implementing various strategies, for stopping the model during training based on specified
    conditions.
    """
    def __init__(self, min_epoch):
        self.train_losses: List[float] = []
        self.test_losses: List[float] = []
        self.min_epoch = min_epoch

    def append_train_loss(self, loss: float):
        """
        Appends the given loss to the train_losses list.

        :param loss:    The loss to append.
        """
        self.train_losses.append(loss)

    def append_test_loss(self, loss: float):
        """
        Appends the given loss to the test_losses list.

        :param loss:    The loss to append.
        """
        self.test_losses.append(loss)

    def has_converged(self, epoch) -> bool:
        """
        Returns whether the given epoch is converged or not.

        :param epoch:   The current epoch.
        :return:    `True` if the epoch is converged, `False` otherwise.
        """
        return epoch >= self.min_epoch

# imports have to be at this location, otherwise my IDE marks this as circular dependency
from src.training.convergers.sma import SimpleMovingAverageConverger
from src.training.convergers.rel_loss import RelativeLossConverger

def get_converger(converger_name: str, min_epoch=0) -> Converger:
    """
    Returns a concrete Converger instance based on the configuration.

    :param converger_name:  The name of the converger to select.
    :param min_epoch:       The minimum number of epochs to train for.
    :return:    The concrete Converger instance.
    """
    match converger_name:
        case "SMA":
            return SimpleMovingAverageConverger(min_epoch)
        case "RelLoss":
            return RelativeLossConverger(min_epoch)
        case _:
            return Converger(min_epoch)