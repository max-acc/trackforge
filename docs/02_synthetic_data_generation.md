# TrackForge | Synthetic Data Generation

As this project first of all resembles a proof of concept, I generated fully **synthetic events**.

## Why Synthetic Data?

* Full Control over ground truth
* Easy to experiment with different noise levels, number of tracks, detector geometry
* Generate as many events as needed for training -> especially valuable when working on limited hardware
* Perfect for understanding the problem from first principles

## Helix Track Generation

After a collision, particles follow **helical trajectories** in a uniform magnetic field. The generator implements this
physics:

* Random intial angle
* Random charges sign -> direction of rotation
* Random radius -> inversely related to momentum: smaller radius, lower momentum
* Hits are only recorded at discrete detector layer `z`-positions

**Noise:** Guassian noise is added in the `x-y` plane. The `z` coordinate is assumed known percisely, as _we_ 
constructed the detector.

## Event Composition

Each synthetic event contains:

* Multiple helical tracks
* Uniformly distributed **noise hits** on the detector layers

Noise is generated directly on the clyndrical surface.

## Features & Design Choices

* Using (`r`, `cos phi`, `sin phi`, `z`) instead of raw (`x`, `y`, `z`) helps with periodic nature. More about that in 
the [Introduction](01_introduction.md)
* This representation makes geometric reasoning easier
* These assumptions are easily justifyable, as they can be calculated from the detector geometry for cheap

## Configrability

All parameters live in `YAML` configs. This makes it easy to create varying datasets for testing the models capabilties.

--- 

**Next:** See [Models](04_models.md).
