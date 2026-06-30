# TrackForge | Evaluation

## Metrics

Currently implemented are the following metrics:

* Accuracy
* Precision
* Recall _-> this is currently the main metric to focus on, as missing real track segments is costly_
* Training / validation loss curves

## Current Results

**GCN + Simple MLP** and **GAT + Simple MLP** both suffer from the imbalance issues described in [Models](04_models.md).

## What Went Well

* End-to-End pipeline works
* Modular design makes swapping encoders/classifiers easy
* Geometric features + learned embeddings si a strong hybrid approach
* Convergence strategies implemented and selectable

## Next Steps & Future Work

* Better imbalance handling
* Track building from predicted edges and therefore also ambiguity resolution
* More realistic data
* C++/CUDA accelaration for larger events
* Hyperparameter adjustments
* Comparison against baseline classical algorithms on same data
