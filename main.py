from src.data.dataloader import get_dataloaders
from src.models.simple_edge_classifier import SimpleEdgeClassifier as EdgeClassifier
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
torch.manual_seed(base_config.get_seed())
np.random.seed(base_config.get_seed())

# --- dataset setup
training_config = TrainingConfig(TRAINING_CONFIG)

train_loader, _, test_loader = get_dataloaders(TRAINING_CONFIG)

# --- model setup
model           = EdgeClassifier(MODEL_CONFIG)
optimizer       = torch.optim.Adam(model.parameters(), training_config.get_learning_rate())

# --- Training ---------------------------------------------------------------------------------------------------------
train_losses = []
test_losses = []

for epoch in range(training_config.get_max_epoch()):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = torch.nn.functional.mse_loss(torch.sigmoid(out), batch.y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    train_losses.append(total_train_loss / len(train_loader))

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            loss = torch.nn.functional.mse_loss(torch.sigmoid(out), batch.y)
            total_test_loss += loss.item()

    test_losses.append(total_test_loss / len(test_loader))

    print(f'Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader):.4f}, Test Loss: {total_test_loss / len(test_loader):.4f}')

    if epoch >= 5 and abs(train_losses[-2] - train_losses[-1]) / train_losses[-1] < 0.01:
        break

# --- Evaluation -------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(train_losses)+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()