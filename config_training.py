from configs.config import Config

class TrainingConfig(Config):
    def __init__(self, config_path):
        super().__init__(config_path)

    def get_dataset_split(self, size):
        len = int(size * self.config['training']['training_split'])
        return [len, size - len]

    def get_batch_size(self) -> int:
        return int(self.config['training']['batch_size'])

    def get_max_epoch(self) -> int:
        return int(self.config['training']['max_epochs'])