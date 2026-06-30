"""
Module for implementing the Edge Classifier selection given a configuration.
"""

from src.models.edge_classifier.simple_edge_classifier import get_model as simple_classifier

def get_edge_classifier(model_config, in_features):
    """
    Select the edge classifier based on the model config.

    :param model_config:    The model configuration.
    :param in_features:     The number of input features.
    :return:    The edge classifier.
    """

    match model_config['name']:
        case "SimpleEdgeClassifier":
            return simple_classifier(model_config, in_features)
        case _:
            raise NotImplementedError("Edge Classifier model not implemented!")
