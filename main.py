import torch
import numpy as np

from src.data.dataloader import get_dataloaders, compute_pos_weight
from src.evaluation.metric_dc import MetricList
from src.evaluation.metrics import evaluate
from src.evaluation.plots import plot_evaluations
from src.models.simple_edge_classifier import SimpleEdgeClassifier as EdgeClassifier
from src.training.converger import get_converger
from configs.config import BaseConfig
from configs.training.config import TrainingConfig

# --- Configs ----------------------------------------------------------------------------------------------------------
BASE_CONFIG     = "configs/default.yaml"
TRAINING_CONFIG = "configs/training/training.yaml"
MODEL_CONFIG    = "configs/model/simple_gnn.yaml"

# --- Setup-------------------------------------------------------------------------------------------------------------
#
base_config = BaseConfig(BASE_CONFIG)

np.random.seed(42)

for _ in range(1):
    seed = np.random.randint(0, 100000)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- dataset setup
    training_config = TrainingConfig(TRAINING_CONFIG)

    train_loader, _, test_loader = get_dataloaders(TRAINING_CONFIG, seed)

    # --- model setup
    model           = EdgeClassifier(MODEL_CONFIG)
    optimizer       = torch.optim.Adam(model.parameters(), training_config.get_learning_rate())
    converger       = get_converger(training_config.get_converger(), training_config.get_min_epoch())

    # --- Training ---------------------------------------------------------------------------------------------------------
    train_losses = []
    test_losses = []
    metrics: MetricList = MetricList()

    pos_weight = compute_pos_weight(train_loader.dataset)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(training_config.get_max_epoch()):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
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
                loss = criterion(out, batch.y.float())
                total_test_loss += loss.item()

        metrics.append(evaluate(model, test_loader))

        converger.append_test_loss(total_test_loss / len(test_loader))

        print(f'Epoch {epoch + 1}, Train Loss: {converger.train_losses[-1]:.4f}, Test Loss: {converger.test_losses[-1]:.4f}')

        if converger.has_converged(epoch):
            break

    # --- Evaluation ---------------------------------------------------------------------------------------------------

    plot_evaluations(converger.train_losses, converger.test_losses, metrics)