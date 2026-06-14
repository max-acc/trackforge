from src.data.dataset import SyntheticDataset
from src.models.simple_edge_classifier import SimpleEdgeClassifier as EdgeClassifier
import torch
from torch_geometric.loader import DataLoader

from config_training import TrainingConfig

import matplotlib.pyplot as plt

# --- Setup-------------------------------------------------------------------------------------------------------------

dataset = SyntheticDataset(root='data/synthetic')
training_config = TrainingConfig("configs/training/training.yaml")

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    training_config.get_dataset_split(len(dataset))
)

train_loader    = DataLoader(train_dataset, batch_size=training_config.get_batch_size(), shuffle=True)
test_loader     = DataLoader(test_dataset, batch_size=training_config.get_batch_size(), shuffle=False)

model = EdgeClassifier(node_features=4, hidden_dim=5, num_layers=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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