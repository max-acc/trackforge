from configs.training.config import TrainingConfig
from src.data.dataset import SyntheticDataset

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

def get_dataloaders(config_path, root="src/data/synthetic"):
    """
    Creates train/val/test dataloaders from the synthetic dataset

    :param config_path: The path ot the training configuration file.
    :param root:        The dir where the data should be stored.
    :return:    train_loader, val_loader, test_loader
    """
    config = TrainingConfig(config_path)

    batch_size = config.get_batch_size()

    full_dataset = SyntheticDataset(root=root)

    # split dataset
    total_size  = len(full_dataset)
    train_size  = int(total_size * config.get_training_split())
    val_size    = int(total_size * config.get_validation_split())
    test_size   = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator()
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Dataset split: {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")

    return train_loader, val_loader, test_loader

def compute_pos_weight(dataset):
    """
    Computes the positive weight by sampling the dataset.

    :param dataset:     The dataset.
    :return:    The ratio between negative and positive edges
    """
    total_pos = 0
    total_neg = 0

    for data in dataset:

        pos = (data.y > 0.5).sum().item()
        neg = (data.y < 0.5).sum().item()

        total_pos += pos
        total_neg += neg

    if total_pos == 0:
        return torch.tensor(1.0)

    return torch.tensor(total_neg / total_pos, dtype=torch.float32)
