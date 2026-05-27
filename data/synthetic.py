import numpy as np
import torch
from torch_geometric.data import Data
import random

def generate_helix_track(num_hits=15, z_range=(-100, 100), r=50, noise_sd=0.2):
    z = np.linspace(z_range[0], z_range[1], num_hits)
    phi = np.linspace(0, 4 * np.pi, num_hits)
    x = r * np.cos(phi) + np.random.normal(0, noise_sd, num_hits)
    y = r * np.sin(phi) + np.random.normal(0, noise_sd, num_hits)
    return np.stack([x, y, z], axis=1)  # (num_hits, 3)

def create_synthetic_event(num_tracks=80, hits_per_track=12, noise_hits= 800):
    all_hits = []
    track_ids = []

    # generate real tracks
    for track_id in range(num_tracks):
        hits = generate_helix_track(num_hits=hits_per_track)
        all_hits.append(hits)
        track_ids.extend([track_id] * hits_per_track)

    # add noise hits
    noise = np.random.uniform(-150, 150, (noise_hits, 3))
    all_hits.append(noise)
    track_ids.extend([-1] * noise_hits)

    hits = np.vstack(all_hits)
    track_ids = np.array(track_ids)

    # convert coordinates
    r = np.sqrt(hits[:, 0]**2 + hits[:, 1]**2)
    phi = np.arctan2(hits[:, 1], hits[:, 0])
    z = hits[:, 2]
    hit_features = np.stack([r, phi, z], axis=1)

    return hit_features, track_ids