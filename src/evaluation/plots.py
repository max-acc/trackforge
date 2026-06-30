"""
Module for plotting training evaluations.
"""

from datetime import datetime
import matplotlib.pyplot as plt

from src.evaluation.metric_dc import MetricList


def plot_evaluations(train_losses: list[float],
                     test_losses: list[float],
                     metrics: MetricList) -> None:
    """
    Plots evaluations for the model training.

    :param train_losses:    A list containg the training losses.
    :param test_losses:     A list containg the testing losses.
    :param metrics:         A `MetricList` containg the metrics of the training
    """
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
