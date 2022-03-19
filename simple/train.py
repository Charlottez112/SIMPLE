from __future__ import annotations

import argparse

import torch
import higher
import torch.nn as nn
from torch.utils.data import DataLoader

from .loss import task_loss, noether_loss_approx, noether_loss_exact
from .util import iter_frames


def train_one_epoch(
    f: nn.Module,
    g_list: list[nn.Module],
    loader: DataLoader,
    outer_optimizer: torch.optim.Optimizer,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter,
    args: argparse.Namespace,
):  
    """Train the model for one epoch."""
    # Select Noether loss function.
    if args.conserve_quantity == "approx":
        noether_loss_func = noether_loss_approx
    elif args.conserve_quantity == "exact":
        noether_loss_func = noether_loss_exact
    else:
        raise ValueError("--conserve-quantity must be 'approx' or 'exact'.")

    
    # Set models to training mode.
    device = torch.device(args.device)
    f.train()
    for g in g_list:
        g.train()

    # Train.
    for i, data in enumerate(loader):
        print(f"Batch {i}")

        # Unpack data.
        sim_position: torch.Tensor = data["position"]  # [B, F, N, 3]
        sim_velocity: torch.Tensor = data["velocity"]  # [B, F, N, 3]
        # sim_temperature: torch.Tensor = data["temperature"]  # [B, F]
        sim_boxdim: torch.Tensor = data["boxdim"]  # [B, F]

        # Initialize the inner optimizer (the one that does the tailoring)
        inner_optimizer = torch.optim.Adam(params=f.parameters(), lr=args.inner_lr)
        with higher.innerloop_ctx(f, inner_optimizer, copy_initial_weights=False) as (
            func_f,
            diff_inner_optimizer,
        ):

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
            writer.add_scalar('Loss/neother', noether_loss.item(), epoch)

            # Iterate through timesteps again to compute task losses.
            task_losses = []
            task_losses_pos = []
            task_losses_vel = []
            curr_iter = iter_frames(sim_position, sim_velocity, sim_boxdim)
            label_iter = iter_frames(sim_position, sim_velocity, sim_boxdim, skip=1)
            for current_state, next_state in zip(curr_iter, label_iter):
                # Unpack data.
                position, velocity, boxdim = current_state
                label = torch.concat([next_state[0], next_state[1]], dim=2)

                # Compute the next state prediction.
                next_state_pred = func_f(position, velocity, boxdim)
                task_loss_total, task_loss_pos, task_loss_vel =task_loss(next_state_pred, label)
                writer.add_scalar('Loss/task/position_step', task_loss_pos)
                writer.add_scalar('Loss/task/velocity_step', task_loss_vel)
                task_losses.append(task_loss_total)
                task_losses_pos.append(task_loss_pos)
                task_losses_vel.append(task_loss_vel)

            # Compute the outer loop gradients (including Hessians).
            mean_loss = torch.stack(task_losses).mean()
            mean_loss.backward()
            writer.add_scalar('Loss/task/total_mean', mean_loss.item(), epoch)
            
            mean_pos_loss = torch.stack(task_losses_pos).mean()
            mean_vel_loss = torch.stack(task_losses_vel).mean()
            writer.add_scalar('Loss/task/position_mean', mean_pos_loss.item(), epoch)
            writer.add_scalar('Loss/task/velocity_mean',mean_vel_loss.item(), epoch)
            writer.flush()

        # Update the state predictor and the quantity predictos.
        outer_optimizer.step()

        # Reset gradients.
        outer_optimizer.zero_grad(set_to_none=True)
