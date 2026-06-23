from datetime import datetime

from src.data.dataloader import get_dataloaders, compute_pos_weight
from src.models.simple_edge_classifier import SimpleEdgeClassifier as EdgeClassifier
from src.training.converger import get_converger
from configs.config import BaseConfig
from configs.training.config import TrainingConfig

import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Configs ----------------------------------------------------------------------------------------------------------
BASE_CONFIG     = "configs/default.yaml"
TRAINING_CONFIG = "configs/training/training.yaml"
MODEL_CONFIG    = "configs/model/simple_gnn.yaml"

# --- Setup-------------------------------------------------------------------------------------------------------------
#
base_config = BaseConfig(BASE_CONFIG)

np.random.seed(42)

for _ in range(10):
    seed = np.random.randint(0, 100000)

    torch.manual_seed(seed)#base_config.get_seed())
    np.random.seed(seed)#base_config.get_seed())

    # --- dataset setup
    training_config = TrainingConfig(TRAINING_CONFIG)

    train_loader, _, test_loader = get_dataloaders(TRAINING_CONFIG)

    # --- model setup
    model           = EdgeClassifier(MODEL_CONFIG)
    optimizer       = torch.optim.Adam(model.parameters(), training_config.get_learning_rate())
    converger       = get_converger(training_config.get_converger())

    # --- Training ---------------------------------------------------------------------------------------------------------
    train_losses = []
    test_losses = []

    pos_weight = compute_pos_weight(train_loader.dataset)
    pos_weight_test = compute_pos_weight(test_loader.dataset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_test = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_test)

    for epoch in range(training_config.get_max_epoch()):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            # loss = torch.nn.functional.mse_loss(torch.sigmoid(out), batch.y)
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        converger.append_train_loss(total_train_loss / len(train_loader))

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                # loss = torch.nn.functional.mse_loss(torch.sigmoid(out), batch.y)
                loss = criterion_test(out, batch.y.float())
                total_test_loss += loss.item()

        converger.append_test_loss(total_test_loss / len(test_loader))

        print(f'Epoch {epoch + 1}, Train Loss: {converger.train_losses[-1]:.4f}, Test Loss: {converger.test_losses[-1]:.4f}')

        if converger.has_converged(epoch):
            break

    # --- Evaluation -------------------------------------------------------------------------------------------------------
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(converger.train_losses)+1), converger.train_losses, label='Train Loss')
    plt.plot(range(1, len(converger.train_losses)+1), converger.test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f"{datetime.now()}.png", dpi=300, bbox_inches='tight')
    #plt.show()