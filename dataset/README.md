# Simulation dataset

We built a dataset of 1,625 MD simulations of a system with 4096 particles. Due to its size, we only put one simulation instance as a sample data point in this repo.

Read the simulation data:

```python
sim = np.load("sample.npz")
```

One simulation data point has three parts:

```python
sim["position"].shape  # (300, 4096, 3)
sim["velocity"].shape  # (300, 4096, 3)
sim["boxdim"].shape    # (300,)
```

where 300 is the number of sampled frames in the simulation and 4096 is the number of particles.
We simulated a 3D system, and thus `position` and `velocity` are 3D vectors.
`boxdim` is the dimension of the simulation box where particles reside.
