"""
Module specifying and generating a synthetic dataset.
"""

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from synthetic import create_synthetic_event
from configs.data.config import SyntheticDataConfig

class SyntheticDataset(InMemoryDataset):
    """
    Dataset modeling synthetic data.
    """

    def __init__(self, root, config_path="./configs/data/synthetic.yaml", transform=None, pre_transform=None):
        self.config = SyntheticDataConfig(config_path)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['synthetic_data.pt']

    def process(self):
        data_list = []

        # initialize event data -> so the config has to be called only once
        num_events = self.config.get_num_events()

        for _ in tqdm(range(num_events), desc="Generating synthetic events",
                      unit="event", ncols=100):

            hits, track_ids = create_synthetic_event(self.config)

            # build graph -> connect hits, that are close in phi and z
            edge_index = self.build_graph(hits)

            # edge labels -> 1 if same track, 0 otherwise
            edge_labels = self.get_edge_labels(edge_index, track_ids)

            # node features
            x = torch.tensor(hits, dtype=torch.float)

            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-8
            x = (x - mean) / std

            # all params have to be tensors
            data = Data(
                x=x,
                edge_index=edge_index,
                y=edge_labels,
                track_id=torch.tensor(track_ids, dtype=torch.long)
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def build_graph(self, hits, phi_threshold=1):
        """
        Construct a graph based on hit data.

        :param hits:            The hits to construct the graph of.
        :param phi_threshold:   The phi threshold to use to construct the graph.
        :return:    The undirected hit graph.
        """

        cos_phi = hits[:, 1]
        sin_phi = hits[:, 2]
        z = hits[:, 3]

        phi = np.arctan2(sin_phi, cos_phi)

        z_positions = np.array(self.config.get_noise_config()['z_positions'])
        layer = np.argmin(np.abs(z[:, None] - z_positions[None, :]), axis=1)

        num_layers = len(z_positions)
        all_edges: list = []

        for k in range(num_layers - 1):
            idx_k = np.where(layer == k)[0]
            idx_k1 = np.where(layer == k + 1)[0]

            if len(idx_k) == 0 or len(idx_k1) == 0:
                continue

            d_phi = np.abs(phi[idx_k][:, None] - phi[idx_k1][None, :])
            d_phi = np.minimum(d_phi, 2 * np.pi - d_phi)

            row_i, col_j = np.where(d_phi < phi_threshold)

            if len(row_i) == 0:
                continue

            src = idx_k[row_i]
            dst = idx_k1[col_j]

            all_edges.append(np.stack([src, dst], axis=1))

        if not all_edges:
            return torch.empty((2, 0), dtype=torch.long)

        edges = np.vstack(all_edges)
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        return to_undirected(edge_index)

    def get_edge_labels(self, edge_index, track_ids):
        """
        Labels edges based on the ground truth, e.g. edges connecting nodes with the same track id belong are true
        edges.

        :param edge_index:  The edge indices.
        :param track_ids:   The track IDs.
        :return:    The edge labels for each edge.
        """

        src, dst = edge_index

        if isinstance(track_ids, np.ndarray):
            track_ids = torch.tensor(track_ids, dtype=torch.long)

        # compute labels
        same_track = track_ids[src] == track_ids[dst]
        not_noise = track_ids[src] != -1
        labels = (same_track & not_noise).float()

        return torch.tensor(labels, dtype=torch.float32)
