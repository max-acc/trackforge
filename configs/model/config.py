from configs.config import Config

class ModelConfig(Config):
    """
    Configuration for the Graph Neural Network (GNN) model.

    Contains hyperparameters defining the architecture of the GNN model.
    """
    def __init__(self, config_path: str):
        super().__init__(config_path)

    def get_node_features(self) -> int:
        """
        Return the number of input features per node (hit).

        :return:
        """
        return int(self.config['model']['node_features'])

    def get_hidden_dim(self) -> int:
        """
        Return the hidden dimension size of the GNN layers.

        :return:
        """
        return int(self.config['model']['hidden_dim'])

    def get_num_layers(self) -> int:
        """
        Return the number of message-passing layers in the GNN.

        :return:
        """
        return int(self.config['model']['num_layers'])