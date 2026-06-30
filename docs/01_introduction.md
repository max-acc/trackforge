# TrackForge | Problem Statement

In High Energy Particle Physics (HEP), particles are smashed together and one tries to understand what happened by
loooking at the _debris_ that remained.\
The key challenge: **Reconstruct Particle Trajectories** from discrete detector hits.

## The Physics

Two particle beams are fired on each other, so that the particles inside those beams collide. The resultic particles 
then travel through a magnetic field following a **helical trajectory** (Lorentz Force).\
Detectors are build in cylindrical layers, so hits appear at known `z` positions. In cylindrical coordinates (`r`, 
`phi`, `z`), real track segments have consistent geometric relationships (small `delta r`, `delta phi`, stable slope).

## Introduction to Particle Track Reconstruction

This leads to the key challenge: **How can we use this information, to reconstruct those particle trajectories?**

For reference, a traditional pipeline looks something like this: 

1. **Hit Detection** \
Charged particles ionize the detector material, leaving measurable signals, called hits, in the tracking detectors. 
Those hits are then being clustered.
2. **Track Finding** \
Group hits, that are believed to belong to the same track. Additional hits along the trajactory are associated with 
seeded tracks to extend and refine track candidates. Some popular approaches are the following:
   1. **Seed** \
   Start with 2 to 3 hits near the beam, that look consistent. Use the Kalman filter method, to predict where the next 
   hit could be, based on position, direction and curvature. Then search for hits close to this prediction.
   Then update the track estimation with teh new hit, and repeat outwards. This approach handles measurements and 
   multiple scattering in a statistical optimal way. However this approach is highly sequential, as each track is built 
   at time. Also errors propagate throughout the track estimation.
   2. **Computer Vision** \
   The hits in the detector space are transformed into a curve in the parameter space. This results in hits that belong
   to the same track producing curves that intersect at one point in parameter space. Therefore tracks can be found by
   peaks in parameter space. This approach is better parallelizable than the previous one (and 
   therefore faster). However, it degrades badly with a high hit density, as the parameter space gest crowded with false 
   peaks.
   3. **Cellular Automaton** \
   Constructing short track segments from two segments that are compatible if they are geometrically consistent, that 
   is, similiar in direction with a small angle in between).
3. **Track Fitting** \
Fitting a curve to the hits, that is, determining an optimal trajectory, that best describes the 
detected hits, while accounting for uncertainty, by obtraining momentum.
6. **Ambiguity Resolution** \
Candidate tracks are compared, and most likely trajectories are selected.
7. **Vertex Reconstruction** \
Tracks are extrapolated to identify the points where they originate.

The attentive reader might recognize some issues that naturally arise in this pipeline. However, the major problems I 
want to focus on concerns Track Finding. Manually finding initial track candidates obviously introduces bias and manual 
work. It also limits the approach to one detector architecture. As soon as the detector changes, we have to find other 
parameters for our initial calculations.

Another problem those classical approaches struggle with is high hit density. With a higher number of hits, the number
of possible hit pairs explodes. That is, the hits scale somewhat badly.

In the following we will mainly focus on tracking, as this seems to be the harder problem to solve.
However, I am currently considering widening the current approach to track fitting, with the reasoning for ambiguity
resolution. I would assume that a model _could_ learn what complete trajectories look like, rather than just comparing
track hits only on a local level, that is, evaluating just a single candidate pair.

## Problem Modeling | Particle Tracking

In the following I will elaborate how to transform this problem into a another structure, that might be benefitial to 
solving particle tracking, in this case a **Graph**.

A Graph can be formally described as pair of Vertices/Nodes and Edges `G=(V,E)`, where an edge can be seen as the
connection between Vertices (`E=VxV`). Nodes and Edges can have properties/features, that contain information. There are
various problems that can be modeled as a Graph, and in the following it is assumed to be common knowledge (we will
not get that deep into Graph Theory (at least for now)).

Just imagine the data received after a collision event. After all assumptions we made previously, hit data for each
detector is received. This hit data contains information about where exactly the hits occured on the detector,
something like `x` and `y` coordinates. It is also known on which detector the hit occured, therefore we can infer a `z`
coordinate (based on the detector geometry _we_ deployed). Those coordinates can be written down relative to the
collision points of the particle beams, therefore they can modelled in a cylindrical coordinate system (`r`, `phi`, 
`z`).

> [!Note]
> This approach makes the assumption that detector hits a point nature. However, real expierment data contains multiple
> pixel hits for each detector hit, as there are individual pixel that track the actual hit.
> 
> There are two things that have to be kept in mind:
> 
> 1. It is probably easier, as there is less ambiguity in the data
> 2. It is probably also harder, as we do not have shape information about a single hit.
> 
> Future approaches may consider those caviats.

> [!Note]
> With some foresight, the coordinates will be modeled and used in the following in four dimensions, as (`r`, 
> `cos(phi)`, `sin(phi)`, `z`). 
> 
> This is due to the fact, that points that are geometrically _close_ to each other, lets say two points with equal `r`
> and `z`, the first point is at `phi=0.01*pi` and the second one at `phi=1.99*pi`. Obviously they are _close_ to 
> another, however the only information available is `phi` which leads to the assumption that those points are _far_
> apart. This is hard for the model to understand, and using `sin` and `cos` helps mitigate this problem.

The problem can then be modelled as **edge classification on a graph**:
* **Nodes** = detector hits (features: `r`, `cos phi`, `sin phi`, `z`)
* **Edges** = candidate pairs between nodes
  * **Label** = 1 if hits belong to the same track, 0 otherwise
* Goal: Learn which edges form consistent track segments, e.g. connected components become full tracks

This makes the implicit graph structure of classical mehtods **explicit and learnable**. GNNs can discover geometric
compatebility rules via message passing instead of fragile hand-tuning (usually I do not like such a strong
assumption without proof, however, in this case it might be fine).

--- 

**Next:** See [Synthetic Data Generation](02_synthetic_data_generation.md).
