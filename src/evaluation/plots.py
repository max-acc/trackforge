from datetime import datetime
from typing import List

import matplotlib.pyplot as plt

from src.evaluation.metric_dc import MetricList


def plot_evaluations(train_losses: List[float],
                     test_losses: List[float],
                     metrics: MetricList):
    file_name = f'{datetime.now()}_'

    # --- plot accuracy, recall and precision
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metrics) + 1), metrics.get_accuracy, label='Accuracy')
    plt.plot(range(1, len(metrics) + 1), metrics.get_recall, label='Recall')
    plt.plot(range(1, len(metrics) + 1), metrics.get_precision, label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.ylim(0,1)
    plt.legend()
    plt.savefig("outputs/" + file_name + "arp.png", dpi=300, bbox_inches='tight')

    # --- plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(train_losses) + 1), test_losses, label='Test Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("outputs/" + file_name + "loss.png", dpi=300, bbox_inches='tight')