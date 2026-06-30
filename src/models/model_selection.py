"""
Module for implementing the GNN selection given a configuration.
"""
import torch

from configs.model.config import ModelConfig
from src.models.gat.simple_gat import SimpleGAT
from src.models.gcn.simple_gcn import SimpleGCN

def get_model(config_path: str) -> torch.nn.Module:
    """
    Return the Graph Neural Network model given specified in a config file.

    :param config_path:     The path to the config file.
    :return:    The concrete model if correctly specified.
    """

    match ModelConfig(config_path).get_graph_net_name():
        case "SimpleGCN":
            return SimpleGCN(config_path)
        case "SimpleGAT":
            return SimpleGAT(config_path)
        case _:
            raise NotImplementedError("Graph Neural Network model not implemented!")
