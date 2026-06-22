import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from configs.model import config
from configs.model.config import ModelConfig


class SimpleEdgeClassifier(torch.nn.Module):
    def __init__(self, config_path):
        super().__init__()

        config = ModelConfig(config_path)
        node_features   = config.get_node_features()
        hidden_dim      = config.get_hidden_dim()
        num_layers      = config.get_num_layers()

        # simple gcn layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # edge prediction
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # get embeddings for source and destination
        src, dst = edge_index
        edge_features = torch.cat([
            x[src], x[dst],
            (x[src] - x[dst]).abs()
        ], dim=1)

        # predict edge score
        edge_scores = self.edge_predictor(edge_features).squeeze(-1)

        return edge_scores