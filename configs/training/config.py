from configs.config import Config

from typing import List

class TrainingConfig(Config):
    """
    Configuration for model training hyperparameters.

    Handles dataset splitting, batch size, learning rate, and training duration settings.
    """
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def get_training_split(self) -> float:
        """
        Return the relative training split.

        :return:    The relative training split.
        """
        return self.config['training']['training_split']

    def get_validation_split(self) -> float:
        """
        Return the relative validation split.

        :return:    The relative validation split.
        """
        return self.config['training']['validation_split']

    def get_batch_size(self) -> int:
        """
        Return the batch size used during training.

        :return:    Integer batch size.
        """
        return int(self.config['training']['batch_size'])

    def get_max_epoch(self) -> int:
        """
        Return the maximum number of training epochs.

        :return:    Maximum epochs to train for.
        """
        return int(self.config['training']['max_epochs'])

    def get_min_epoch(self) -> int:
        """
        Return the minimum number of training epochs.

        :return:    Minimum epochs to train for.
        """
        return int(self.config['training']['min_epochs'])

    def get_learning_rate(self) -> float:
        """
        Return the learning rate for the optimizer.

        :return:    Learning rate value.
        """
        return float(self.config['training']['learning_rate'])

    def get_converger(self):
        """
        Return the converger for training stop condition.

        :return:    The name of the stop condition.
        """
        return self.config['training']['converger']