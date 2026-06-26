from datetime import datetime
from typing import List

import matplotlib.pyplot as plt

from src.evaluation.metric_dc import Metric


def plot_evaluations(train_losses: List[float],
                     test_losses: List[float],
                     metrics: List[Metric]):
    file_name = f'{datetime.now()}_'

    # --- plot recall and precision
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics) + 1), [metric.recall for metric in metrics], label='Recall')
    plt.plot(range(1, len(metrics) + 1), [metric.precision for metric in metrics], label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("outputs/" + file_name + "rp.png", dpi=300, bbox_inches='tight')

    # --- plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(train_losses) + 1), test_losses, label='Test Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("outputs/" + file_name + "loss.png", dpi=300, bbox_inches='tight')