"""
Module for specifying a simple MLP for classification tasks.
"""

import torch


def get_model(model_config, in_features: int) -> torch.nn.Sequential:
    """
    Returns a simple edge classifier model.

    :param model_config:    The model configurations in a dictionary.
    :param in_features:     The number of input features.
    :return:    Returns a torch.nn.Sequential model.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, model_config['hidden_dims'][0]),
        torch.nn.BatchNorm1d(model_config['hidden_dims'][0]),
        torch.nn.ReLU(),
        torch.nn.Dropout(model_config['dropout']),
        torch.nn.Linear(model_config['hidden_dims'][0], model_config['hidden_dims'][1]),
        torch.nn.ReLU(),
        torch.nn.Linear(model_config['hidden_dims'][1], 1)
    )
