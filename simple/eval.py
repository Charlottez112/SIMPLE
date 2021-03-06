from __future__ import annotations

import time
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
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
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
    position_losses = []
    velocity_losses = []
    # To track the mean inference latency.
    latencies = []
    for i, data in enumerate(loader):
        step_counter = 0
        print(f"Batch {i}")

        # Track inference latency.
        batch_start_time = time.time()

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

                position = position.to(device)
                velocity = velocity.to(device)
                boxdim = boxdim.to(device)

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
            # Log the initial neother loss
            writer.add_scalar(f'Noether_Loss/Val/{epoch}/initial', noether_loss, i)
            diff_inner_optimizer.step(noether_loss)

            # Re-predict the next states on Tailored State predictor
            noether_embeddings = []
            for position, velocity, boxdim in iter_frames(
                sim_position, sim_velocity, sim_boxdim
            ):  

                position = position.to(device)
                velocity = velocity.to(device)
                boxdim = boxdim.to(device)

                # Compute the next state prediction.
                next_state_pred = func_f(position, velocity, boxdim)

                # Run through all quantity predictors to compute Noether embeddings.
                # NOTE: All quantity predictor modules must return tensors of shape [B, e].
                noether_embedding = torch.concat(
                    [g(next_state_pred) for g in g_list], dim=1
                )
                noether_embeddings.append(noether_embedding)

            
            noether_loss = noether_loss_func(noether_embeddings)
            # Log the final neother loss
            writer.add_scalar(f'Noether_Loss/Val/{epoch}/Final', noether_loss, i)

            # Henceforth are pure inference.
            with torch.no_grad():
                # Iterate through timesteps again to compute task losses.
                # Note that we do not perform inference on the last frame because there is
                # no ground truth state for it's next state.
                curr_iter = iter_frames(sim_position, sim_velocity, sim_boxdim)
                label_iter = iter_frames(sim_position, sim_velocity, sim_boxdim, skip=1)
                for j, (current_state, next_state) in enumerate(zip(curr_iter, label_iter)):
                    # Unpack data.
                    position, velocity, boxdim = current_state
                    position = position.to(device)
                    velocity = velocity.to(device)
                    boxdim = boxdim.to(device)

                    label = torch.concat([next_state[0], next_state[1]], dim=2)
                    label = label.to(device)

                    # Compute the next state prediction.
                    # On the first step, the initial position, velocity, and boxdim are input.
                    # On all subsequent steps, the previous step's position and velocity preds,
                    # plus the current timestep's boxdim (from the user) are input.
                    if j == 0:
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
                    task_loss_total, task_loss_pos, task_loss_vel = task_loss(next_state_pred, label)
                    task_loss_pos = task_loss_pos.cpu().detach()
                    task_loss_vel = task_loss_vel.cpu().detach()
                    
                    writer.add_scalar(f'Loss/Val/{epoch}/batch_{i}/position_step', task_loss_pos, step_counter)
                    writer.add_scalar(f'Loss/Val/{epoch}/batch_{i}/velocity_step', task_loss_vel, step_counter)
                    step_counter += 1
                    
                    task_losses.append(task_loss_total.cpu().detach())
                    position_losses.append(task_loss_pos)
                    velocity_losses.append(task_loss_vel)

                    # Save prediction. This is left in CUDA memory for the next frame prediction.
                    state_preds.append(next_state_pred.detach())

                # Send the second-to-last frame's next frame prediction to CPU.
                state_preds[-1] = state_preds[-1].cpu()
                writer.flush()


        # Track inference latency.
        inference_latency = time.time() - batch_start_time
        latencies.append(inference_latency)
        print(f"Inference latency: {inference_latency}")

    # Compute mean inference latency and print.
    print(f"Mean inference latency: {sum(latencies) / len(latencies)}")
    print(f"Mean inference latency excluding first batch: {sum(latencies[1:]) / len(latencies[1:])}")

    return sum(task_losses) / len(task_losses), sum(position_losses) / len(position_losses), sum(velocity_losses) / len(velocity_losses)

