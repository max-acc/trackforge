import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from .synthetic import create_synthetic_event
from .config_synthetic_data import SyntheticDataConfig

class SyntheticDataset(InMemoryDataset):
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

    def build_graph(self, hits, phi_threshold=0.1, z_threshold=20.0):
        phi = hits[:, 1]
        z = hits[:, 2]

        # TODO: N^2 good enough for small datasets -> has to be optimized
        edges = []
        for i in range(len(hits)):
            for j in range(i+1, len(hits)):
                delta_phi = abs(phi[i] - phi[j])
                delta_phi = min(delta_phi, 2 * np.pi * delta_phi)
                if delta_phi < phi_threshold and abs(z[i] - z[j]) < z_threshold:
                    edges.append([i, j])


        if len(edges) == 0:
            edge_index = torch.empty((2,0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T
        return to_undirected(edge_index)

    def get_edge_labels(self, edge_index, track_ids):
        src, dst = edge_index

        if isinstance(track_ids, np.ndarray):
            track_ids = torch.tensor(track_ids, dtype=torch.long)

        # compute labels
        same_track = track_ids[src] == track_ids[dst]
        not_noise = track_ids[src] != -1
        labels = (same_track & not_noise).float()

        return torch.tensor(labels, dtype=torch.float32)