from __future__ import annotations

import argparse

import torch
import higher
import torch.nn as nn
from torch.utils.data import DataLoader

from .loss import task_loss, noether_loss_approx, noether_loss_exact
from .util import iter_frames


def evaluate(
    f: nn.Module,
    g_list: list[nn.Module],
    loader: DataLoader,
    args: argparse.Namespace,
) -> float:
    """Evaluate the given model."""
    # Select Noether loss function.
    if args.conserve_quantity == "approx":
        noether_loss_func = noether_loss_approx
    elif args.conserve_quantity == "exact":
        noether_loss_func = noether_loss_exact
    else:
        raise ValueError("--conserve-quantity must be 'approx' or 'exact'.")

    device = torch.device(args.device)
    # Set models to evaluation mode.
    f.eval()
    for g in g_list:
        g.eval()

    # Evaluate.
    # Saves all task losses and predicted states.
    task_losses = []
    state_preds = []
    for i, data in enumerate(loader):
        print(f"Batch {i}")

        # Unpack data.
        sim_position: torch.Tensor = data["position"]  # [B, F, N, 3]
        sim_velocity: torch.Tensor = data["velocity"]  # [B, F, N, 3]
        # sim_temperature: torch.Tensor = data["temperature"]  # [B, F]
        sim_boxdim: torch.Tensor = data["boxdim"]  # [B, F]

        # Initialize the inner optimizer (the one that does the tailoring)
        inner_optimizer = torch.optim.Adam(params=f.parameters(), lr=args.inner_lr)
        # Since we're not performing the outer loop updates,
        # we don't need to track higher order gradients.
        with higher.innerloop_ctx(
            f, inner_optimizer, copy_initial_weights=False, track_higher_grads=False
        ) as (func_f, diff_inner_optimizer):

            # Iterate through frames and compute Noether embeddings.
            noether_embeddings = []
            for position, velocity, boxdim in iter_frames(
                sim_position, sim_velocity, sim_boxdim
            ):  

                position.to(device)
                velocity.to(device)
                boxdim.to(device)

                # Compute the next state prediction.
                next_state_pred = func_f(position, velocity, boxdim)

                # Run through all quantity predictors to compute Noether embeddings.
                # NOTE: All quantity predictor modules must return tensors of shape [B, e].
                noether_embedding = torch.concat(
                    [g(next_state_pred) for g in g_list], dim=1
                )
                noether_embeddings.append(noether_embedding)

            # Tailor the state predictor on the Noether loss.
            noether_loss = noether_loss_func(noether_embeddings)
            diff_inner_optimizer.step(noether_loss)

            # Henceforth are pure inference.
            with torch.no_grad():
                # Iterate through timesteps again to compute task losses.
                # Note that we do not perform inference on the last frame because there is
                # no ground truth state for it's next state.
                curr_iter = iter_frames(sim_position, sim_velocity, sim_boxdim)
                label_iter = iter_frames(sim_position, sim_velocity, sim_boxdim, skip=1)
                for i, (current_state, next_state) in enumerate(
                    zip(curr_iter, label_iter)
                ):
                    # Unpack data.
                    position, velocity, boxdim = current_state
                    position.to(device)
                    velocity.to(device)
                    boxdim.to(device)

                    label = torch.concat([next_state[0], next_state[1]], dim=2)
                    label.to(device)

                    # Compute the next state prediction.
                    # On the first step, the initial position, velocity, and boxdim are input.
                    # On all subsequent steps, the previous step's position and velocity preds,
                    # plus the current timestep's boxdim (from the user) are input.
                    if i == 0:
                        next_state_pred = func_f(position, velocity, boxdim)
                    else:
                        prev_state_pred = state_preds[-1]
                        next_state_pred = func_f(
                            prev_state_pred[:, :, 0:3],
                            prev_state_pred[:, :, 3:6],
                            boxdim,
                        )
                        # No need to keep it in CUDA now we're done with it.
                        state_preds[-1] = prev_state_pred.cpu()

                    # Compute and save task loss.
                    task_losses.append(task_loss(next_state_pred, label).cpu().detach())

                    # Save prediction. This is left in CUDA memory for the next frame prediction.
                    state_preds.append(next_state_pred.detach())

                # Send the second-to-last frame's next frame prediction to CPU.
                state_preds[-1] = state_preds[-1].cpu()

    return sum(task_losses) / len(task_losses)
