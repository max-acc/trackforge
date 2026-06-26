import torch

from src.evaluation.metric_dc import Metric


def evaluate(model, loader, threshold=0.5) -> Metric:
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            scores = torch.sigmoid(logits)
            all_scores.append(scores)
            all_labels.append(batch.y)

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    preds = (scores > threshold).astype(float)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    print([tp, fp, fn, tn])

    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0

    return Metric(recall, precision)