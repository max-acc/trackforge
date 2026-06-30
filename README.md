# <h1 align="center">TrackForge</h1>

> Particle Tracks Reconstruction via Graph Neural Networks

_TrackForge_ is a from scratch **Graph Neural Network (GNN) pipeline** for reconstructing particle tracks in high energy 
physics (HEP) detectors. Hits become nodes, candidate connections become edges, and the model learns to classify which
pairs belong to teh same helical trajectory.

Built by a CS major with a Physics minor, this project sits at the intersection of machine learning and particle 
physics. The goal is not to beat state of the art systems but to deeply understand every component, while documenting
the journey (including dead ends).

It is designed to be accessible for others with similar backgrounds who find HEP intersting, and want to start out.

---

### Objectives

| Academic                          | Technical                            | Softwareengineering                      |
|-----------------------------------|--------------------------------------|------------------------------------------|
| Deepen GNN understanding          | GNN + PyG pipeline on synthetic data | Improve Python Proficiency               |
| Deepen particle physics intuition | Hit pairing / edge classification    | Modular design for easy experimentation  |
|                                   |                                      | Explore CUDA / performance-critical code |


### Scope and Limitations
This is **not** an attempt to compete with production track reconstruction systems (e.g., those used at LHC 
experiments). I deliberately avoided deep literature dives early on to develop my own intuition and solutions first.

I do not have formal physics qualifications (yet). Assumptions are clearly stated with reasoning. Performance is modest,
but learning is massive.

### Repository Structure

- **[Problem & Modeling](/docs/01_introduction.md)**
- **[Synthetic Data Generation](/docs/02_synthetic_data_generation.md)**
- **[Models](/docs/04_models.md)**
- **[Training](/docs/05_training.md)**
- **[Evaluation](/docs/06_evaluation.md)**

---

### Current Models

**Graph Encoders**
* Simple GCN
* Simple GAT

**Graph Attention**
* Simple MLP

Everything is configurable via `YAML` files for rapid eexperimentation.

## Run it Yourself

After cloning and `cd`-ing, you can just run `make run` to generate a synthetic dataset and start the model training
and evaluation.

Note that the generated dataset will be used for further runs. You can delete it with `make clear-synthetic`. 
Regarding the dependecies, just try to run the model with small config parameters, and install what you are being
expected to.
