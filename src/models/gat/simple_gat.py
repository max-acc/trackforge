"""
Module implementing a simple Graph Transformer based classifier.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from configs.model.config import ModelConfig
from src.models.model_classifier_selection import get_edge_classifier


class SimpleGAT(torch.nn.Module):
    """

    Components:
        * Graph Transformer Convolutions
        * Batch Normalization
        * Edge Classifier (as specified in config)
    """

    def __init__(self, config_path):
        super().__init__()

        config          = ModelConfig(config_path)
        graph_net       = config.get_graph_net()
        classifier_net  = config.get_classifer_net()

        node_features   = graph_net['node_features']
        num_heads       = graph_net['num_heads']
        latent_features = graph_net['latent_features']
        num_layers      = graph_net['num_layers']
        hidden_dim      = graph_net['hidden_dim']

        # simple gcn layers and batchnorm
        self.convs  = torch.nn.ModuleList()
        self.bns    = torch.nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim * num_heads, heads=num_heads, concat=False))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim * num_heads))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim * num_heads, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * num_heads))

        # edge prediction
        self.edge_predictor = get_edge_classifier(classifier_net, hidden_dim * num_heads * 2 + latent_features)

    def geometric_features(self, raw_x, src, dst):
        """
        Construct geometric features from source and destination nodes.

        :param raw_x:   The raw data.
        :param src:     The source nodes.
        :param dst:     The destination nodes.
        :return:    A tensor of features.
        """

        r_s = raw_x[src, 0]
        r_d = raw_x[dst, 0]
        cos_s = raw_x[src, 1]
        sin_s = raw_x[src, 2]
        cos_d = raw_x[dst, 1]
        sin_d = raw_x[dst, 2]
        z_s = raw_x[src, 3]
        z_d = raw_x[dst, 3]

        dr = r_d - r_s
        dz = z_d - z_s

        dphi = torch.atan2(sin_d * cos_s - cos_d * sin_s,
                           cos_d * cos_s - sin_d * sin_s)

        eps = 1e-4
        dphi_dz = dphi / dz.abs().clamp(min=eps)
        dr_dz = dr / dz.abs().clamp(min=eps)

        features = torch.stack([
            dr,
            dz,
            dphi,
            dphi.cos(),
            dphi.sin(),
            dphi_dz,
            dr_dz,
            r_s,
            r_d,
            z_s,
            z_d,
            (dr ** 2 + dz ** 2).sqrt()
        ], dim=1)

        return features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        raw_x = data.x

        for i, (conv, bns) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = bns(F.relu(x))
            else:
                x = F.relu(x)

        # get embeddings for source and destination
        src, dst = edge_index

        features = self.geometric_features(raw_x, src, dst)
        features = torch.cat([x[src], x[dst], features], dim=1)

        # predict edge score
        edge_scores = self.edge_predictor(features).squeeze(-1)
        return edge_scores
