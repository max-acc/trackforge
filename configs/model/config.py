"""
Module containing a configuration class for the model parameters.
"""
from configs.config import Config

class ModelConfig(Config):
    """
    Configuration for the Graph Neural Network (GNN) model.

    Contains hyperparameters defining the architecture of the GNN model.
    """

    def get_graph_net_name(self) -> str:
        """
        Returns the name of the graph neural network model.

        :return:    The name of the graph neural network model.
        """
        return str(self.config['model']['graph_net'])

    def get_graph_net(self):
        """
        Returns the graph neural network model's parameters.

        :return:    Return a dictionary containing the parameters of the specified graph neural network model.
        """
        match self.config['model']['graph_net']:
            case "SimpleGCN":
                return {
                    'name':             self.config['model']['graph_net'],
                    'node_features':    int(self.config['simple_gcn']['node_features']),
                    'latent_features':  int(self.config['simple_gcn']['latent_features']),
                    'hidden_dim':       int(self.config['simple_gcn']['hidden_dim']),
                }
            case "SimpleGAT":
                return {
                    'name':             self.config['model']['graph_net'],
                    'node_features':    int(self.config['simple_gat']['node_features']),
                    'num_heads':        int(self.config['simple_gat']['num_heads']),
                    'latent_features':  int(self.config['simple_gat']['latent_features']),
                    'num_layers':       int(self.config['simple_gat']['num_layers']),
                    'hidden_dim':       int(self.config['simple_gat']['hidden_dim']),
                }
            case _:
                raise NotImplementedError("Graph Neural Network model not implemented!")


    def get_classifer_net(self):
        """
        Returns the classifier model's parameters.

        :return:    Return a dictionary containing the parameters of the specified classifier network model.
        """
        match self.config['model']['classifier']:
            case "SimpleEdgeClassifier":
                return {
                    'name':         self.config['model']['classifier'],
                    'hidden_dims':  self.config['simple_edge_classifier']['hidden_dims'],
                    'dropout':      int(self.config['simple_edge_classifier']['dropout'])
                }
            case _:
                raise NotImplementedError("Edge Classifier model not implemented!")
