# TrackForge | Training

## Training Loop Overview

The training pipeline follows a standard PyTorch + PyG pattern with several custom components for this project.

### Key Steps per Epoch

1. Forward pass through the chosen GNN encoder + edge classifier
2. Compute `BCEWighLogitsLoss` with a positive weight (necessary due to heavy class imbalance)
3. Backpropagation + Admam Optimizer + ReduceLOnPlateau scheduler
4. Evaluation on test set (metrics + loss)
5. Convergence check

## Custom Components

### Convergence Strategies

Multiple stopping criteria has been implemented, as the standard _fixed epochs_ did not work well with varying dataset
difficulty:

* **BaseConverger:** Minimum epochs
* **Relative Loss:** Stops when the relative change between the last two epochs is small
* **Simple Moving Average (SMA):** Averages the loss over a specified window, and if small enough, stops training
* **Exponential Moving Average (EMA):** Similar to previous, however builds average from all previous losses, which is
computationally cheaper

### Model Selection

Both graph encoder and edge classifier are selected via factory functions driven by YAML configs. This modularity was a
major design goal.

--- 

**Next:** See [Evaluation](06_evaluation.md).
