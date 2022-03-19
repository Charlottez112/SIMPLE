from __future__ import annotations

import torch


def task_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """A function that measures how accurate the state prediction is.

    Args:
        pred (Tensor): Prediction of the next state. Shape: [B, N, 6].
        label (Tensor): Ground truth next state. Shape: [B, N, 6].
    """
    return torch.mean(torch.square(pred - label)), torch.mean(torch.square(pred[:,:,:3] - label[:,:,:3])), torch.mean(torch.square(pred[:,:,3:] - label[:,:,3:]))


def noether_loss_exact(quantities: list[torch.Tensor]) -> torch.Tensor:
    """The Noether loss function that exactly conserves quantities.

    Args:
        quantities (list[Tensor]): List of quantity tensors from each
            time step.

    The first item of the `quantities` list serves as the ground truth
    quantity that all subsequent quantities must be similar to.
    """
    first_state_quantity = quantities[0]
    later_quantities = torch.stack(quantities[1:])
    return torch.mean(torch.square(later_quantities - first_state_quantity))


def noether_loss_approx(quantities: list[torch.Tensor]) -> torch.Tensor:
    """The Noether loss function that approximately conserves quantities.

    Args:
        quantities (list[Tensor]): List of quantity tensors from each
            time step.

    Compares quantities of adjacent time steps and enforces that they
    be similar.
    """
    quants = torch.stack(quantities)
    return torch.mean(torch.square(quants[1:] - quants[:-1]))
