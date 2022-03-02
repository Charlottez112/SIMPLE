from __future__ import annotations

from typing import Generator

import torch


def iter_frames(
    sim_position: torch.Tensor,
    sim_velocity: torch.Tensor,
    sim_boxdim: torch.Tensor,
    skip: int = 0,
) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """A generator that yields subsequent frames of a single simulation.

    The generator will skip (not yield) the first `skip` frames.
    """
    F = sim_position.shape[1]
    positions = sim_position.split(1, dim=1)
    velocities = sim_velocity.split(1, dim=1)
    boxdims = sim_boxdim.split(1, dim=1)

    for frame in range(skip, F):
        yield (
            positions[frame].squeeze(1),
            velocities[frame].squeeze(1),
            boxdims[frame].squeeze(1),
        )
