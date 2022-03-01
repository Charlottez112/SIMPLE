from __future__ import annotations

import torch
import torch.nn as nn


class StatePredictor(nn.Module):
    """Model that predicts the next state in simulation.

    Input is the current state (position & velocity) matrix with
    the initial configuration vector. Output is the next predicted
    state matrix.

    For each particle, the k nearest neighbor particles are identified
    based on the particles' position. Then, only the k neighbors are
    used to predict the target particle's next position and velocity
    by a feed-forward neural network.
    """

    def __init__(self, num_neighbors: int = 10) -> None:
        """Intialize the model.

        Args:
            num_neighbors (int): The number of nearest neighbors to
                consider when predicting the particle's next state.
        """
        super().__init__()

        self.num_neighbors = num_neighbors

        # Input components concatenated for one timestep:
        # - Velocity vectors of k nearest neighbors and self
        # - Position offset vectors of k nearest neighbors
        # - The dimension of the box
        len_input = 3 * (num_neighbors + 1) + 3 * num_neighbors + 1

        # Three-layer feed forward network as the next state predictor.
        self.layers = nn.Sequential(
            nn.Linear(len_input, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

    def forward(
        self,
        position: torch.Tensor,  # [B, N, 3]
        velocity: torch.Tensor,  # [B, N, 3]
        boxdim: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Predict the next state of all particles."""
        # Unpack data.
        B, N, _ = position.shape
        k = self.num_neighbors

        # Compute pairwise distances.
        # NOTE: This uses O(N^2) memory. Might have to loop over particles if
        #       memory blows up.
        distances = torch.cdist(position, position)  # [B, N, N]

        # Find the k nearest neighbors.
        # k + 1 because this will always include the particle itself.
        # Sorted to ease slicing off the self-index, and to give the model
        # a more structured input (i.e. particles are sorted in decreasing
        # order of influence).
        # Shape: [B, N, k+1]
        nn_indices = torch.topk(distances, k=k + 1, largest=False, sorted=True).indices

        # Delete pairwise distnace tensor since it's no longer needed.
        del distances

        # Gather nearest neighbor velocities for each particle.
        # Shape: [B, N, k+1, 3]
        expanded_idx = nn_indices.unsqueeze(3).repeat(1, 1, 1, 3)
        velocity_nn = velocity.unsqueeze(1).repeat(1, N, 1, 1).gather(2, expanded_idx)

        # Gather nearest neighbor positions for each particle.
        # Shape: [B, N, k, 3]
        no_self_idx = expanded_idx[:, :, 1:, :]
        position_nn = position.unsqueeze(1).repeat(1, N, 1, 1).gather(2, no_self_idx)

        # Delete index matrices since they're no longer needed.
        del no_self_idx, expanded_idx, nn_indices

        # Offset only the position vector.
        position_nn_offset = position_nn - position.unsqueeze(2)  # [B, N, k, 3]

        # Concatenate velocity, position, and boxdim vectors.
        # Shape: [B, N, 3*(k+1)+3*k+1]
        current_state = torch.concat(
            [
                velocity_nn.view(B, N, 3 * (k + 1)),
                position_nn_offset.view(B, N, 3 * k),
                boxdim.unsqueeze(1).unsqueeze(2).repeat(1, N, 1),
            ],
            dim=2,
        )
        # Shape: [B*N, 3*(k+1)+3*k+1]
        flattened_current_state = current_state.reshape(B * N, 6 * k + 3 + 1)

        # Run through neural network.
        pred = self.layers(flattened_current_state)  # [B*N, 6]
        unflattened_pred = pred.reshape(B, N, 6)  # [B, N, 6]

        # Apply delta to position. This is our predicted next state.
        unflattened_pred[:, :, :3] += position

        return unflattened_pred


class LearnedQuantityPredictor(nn.Module):
    """Model that predicts the quantity to be conserved."""

    def __init__(self, embedding_dim: int = 8, num_particles: int = 4096) -> None:
        """Intialize the model.

        Args:
            embedding_dim (int): Dimension of the Noether embedding.
            num_particles (int): Number of particles in the system.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_particles = num_particles

        # Input components concatenated:
        # - Velocity vectors of all particles
        # - Position offset vectors of all particles
        len_input = num_particles * 6

        # Three-layer feed forward network as the quantity predictor.
        self.layers = nn.Sequential(
            nn.Linear(len_input, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

    def forward(
        self, state: torch.Tensor# [B, N, 6]
    ) -> torch.Tensor:
        """Predict the conserved quantity."""
        # Unpack data
        B, N, _ = state.shape

        # Run through the neural network.
        return self.layers(state.view(B, 6*N))  # [B, e]


class TemperaturePredictor(nn.Module):
    """Simple module that computes the temperature of each state.

    The computed value is actually the mean of all particle's velocity.
    The actual temperature cannot be computed since we don't have access to
    the mass of each particle.
    """

    def forward(
        self,
        state: torch.Tensor,  # [B, N, 6]
    ) -> torch.Tensor:
        """Computes the temperature of the given state."""
        velocities = torch.linalg.vector_norm(state[:, :, 3:], dim=2)  # [B, N]
        return torch.mean(velocities, dim=1, keepdim=True)  # [B, 1]
