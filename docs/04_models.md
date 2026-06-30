# TrackForge | Models

The hit pairing problem is framed as **edge classification on a graph**: detecctor hits become nodes, candiate hit pairs
become edges, and the network to output a score per edge for _is a real segment of a particle trajectory, or is it
noise_. The pipeline is split into two concatenated states so that either half can be swaped independently (see [Model 
Selection](#model-selection) below):

1. A **graph encoder** that turns each node's raw data into a learned embedding by passing messages along the candidate
edge graph.
2. An **edge classifier** that takes a per edge feature vector and outputs a single logit per edge.

## Shared Design: Why Mix Learned Embeddings with Engineered Features?

The currently implemented Graph Models compute features form raw node coordinates before concatenating with the GNN 
learned embeddings.

The reasoning for not relying purely on the learned embeddings: in a cylindrical detector, a real track segment is 
constrained by physics to look roughly like a a mildly curved line in the r-z and r-phi projects. Quantities like the 
delta radius and delta phi, and the local slope are cheap, and already physically meaningful discriminators. Handing 
them to the classifier directly gives it a strong prior _shortcut_ instead of forcing the GNN to rediscover baisc
detetector geometry from scratch with limited data and limited depth.

## Graph Models

The models listed below increase in complexity in their listed sequence.

### Graph Convolutional Network (GCN)

This implementation stacks Graph Convolution Layers, each followed by BatchNorm and ReLU, then concatenates the final
node embeddings of both edge endpoints with the geometric features above before passing the result to the configured
edge classifier. GCN was chosen as the first baseline because mean/sum aggregation is simple, well understood, and 
cheap.

### Graph Transformer Network (GAT)

This implementation follow the identical structure to the GCN above, but replaces the Graph Convolutions with Graph
Transformer Convolutions. Where GCN averages all neighbors uniformly, GAT learns per edge attention weights, wich is 
attractive for this problem spedifically bechause detector regions can be locally dense, and attention gives the network
a mechanism to down weight implauible neighbors instead of treating them all equally

## Edge Classifier Models

The models listed below increase in complexity in their listed sequence.

### Simple Edge Classifier

This implemention returns a small MLP:\
`Linear -> ReLU -> BatchNorm -> Dropout -> Linear -> ReLU -> Linear`,\
taking in the concatenated node embedding and the calculated features vector and producing one raw logit per edge. This
is intentionally the simplest possible second stage, so taht any performance difference between GCN and GAT in the 
comparison table below can be attributed to the encoder rather than the classifier.

> [!Note]
> It is important to use `BCEWithLogitLoss` as a loss function in the training loop, as this is numerically more stable
> than using `Sigmoid` and `BCELoss`.

> [!Note]
> The usage of only one Dropout layer after the first Linear layer is intentional, as the regularizes harder near the
> raw, high dimensional input.

## Model Selection

Both graph encoder and edge classifier are selected through small factory functions driven by a config file. This keeps
the two stages orthogonal.

# Model evaluation

The are various metrics used for evaluating the different model combinations. \
However, in this overview we will use Recall as metric, as this is the most interesing metric given the current state of
the evaluation metrics. You can read more on model evaluation [here](06_evaluation.md).

> [!WARNING]
> Test data already exists, however I did not yet find the time to include it.

| | GCN   | GAT  |
|---|-------|------|
|Simple Edge Classifier|  | |

> [!INFO]
> The capacity of the models itself is hard to compare. Therefore they are compared based on the largest possible model
> capacity that can be run on the same limited hardware.\
> Training parameters and Dataset stays the same.

### Data Configuration

| | |

### Training Configuration

| Training Split | Validation Split | Test Split | Batch Size | Learning Rate | Minimum/Maximimum Epochs | Converger |
|----------------|------------------|------------|------------|---------------|--------------------------|-----------| 
| 80%            | 0%            | 20%    | 64         | 20/100        | 0.001                    | RelLoss   |


## GCN and Simple Edge Classifier

### Model Parameters
**Graph Net**

| Model | Node Features | Latent Features | Layers | Hidden Dimensionality | Dropout | 
|-------|---------------|-----------------|--------|-----------------------|---------|
| GCN   | 4             | 12              | 5      | 64                    | -       |
| MLP   | -             | -               | 3      | [128, 64]             | 0.2     |



### Evaluation

[LossImage]
[Accuracy, Precision, Recall Curve Image]


## GCN and Simple Edge Classifier

### Model Parameters

| Node Features | Latent Features | Layers  | Hidden Dimensionality | Heads  | Dropout |
|---------------|-----------------|---------|-----------------------|--------|---------|
| 4             | 12              | 5       | 64                    | 4      | -       |
| MLP           | -               | 3       | [128, 64]             | -      | 0.2     |

### Evaluation
[LossImage]
[Accuracy, Precision, Recall Curve Image]

## Challenges and Lessons Learned

A major pitfall I encountered:

* **Extreme class imbalance:** True track edges are rare compared to noise/false pairs
* Early models often **predicted all negatives/positives**

This led to terrible recall/precision initially. I only realized how it was when comparing to results from other papers.

**Mitigation tried**
* `BCEWithLogitsLoss` + `pos_weight`
* Gemotric Feature Engineering (mentioned above)
* Attention (GAT) -> did not really help
* Convergence Heuristics (more about that in [Evaluation](06_evaluation.md))

--- 

**Next:** See [Training](05_training.md).
