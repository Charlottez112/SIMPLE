from __future__ import annotations

import argparse
import json
import time
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from simple.model import StatePredictor, LearnedQuantityPredictor, TemperaturePredictor
from simple.data import SimulationDataLoader
from simple.train import train_one_epoch
from simple.eval import evaluate


def main(args):
    device = torch.device(args.device)
    # Initialize models.
    f = StatePredictor(args.activation, args.predict_velocity_diff, args.num_neighbors)
    f.to(device)
    g_list: list[nn.Module] = [LearnedQuantityPredictor(args.activation_noether,
                                                        args.embedding_dim),
                               TemperaturePredictor()]
    for g in g_list:
        g.to(device)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs('./Log', exist_ok=True)
    writer = SummaryWriter(f'./Log/{current_time}')

    # Initialize data loaders.
    train_loader = SimulationDataLoader(
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = SimulationDataLoader(
        split="eval",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialize the outer optimizer (the one that backprops from task_loss).
    outer_params = list(f.parameters())
    for g in g_list:
        outer_params.extend(list(g.parameters()))
    outer_optimizer = torch.optim.Adam(params=outer_params, lr=args.outer_lr)

    # Run training.
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        train_one_epoch(f, g_list, train_loader, outer_optimizer, epoch, writer, args)
        val_error_total, val_error_pos, val_error_vel = evaluate(f, g_list, eval_loader, writer, args)
        print(f"validation error: {val_error_total}")
        writer.add_scalar('Val_error/total', val_error_total, epoch)
        writer.add_scalar('Val_error/position', val_error_pos, epoch)
        writer.add_scalar('Val_error/velocity', val_error_vel, epoch)
        writer.flush()
    writer.close()

    os.makedirs(f'./Saved_Models/{current_time}', exist_ok=True)
    torch.save(f.state_dict(), f'./Saved_Models/{current_time}/{f}')
    for g in g_list:
        torch.save(g.state_dict(), f'./Saved_Models/{current_time}/{g}')

    # Save hyperparameters along with the model
    with open(f'./Saved_Models/{current_time}/hyperparameters.json', 'wt') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--outer_lr', default=1e-4, type=float, help='learning rate for outer loop')
    parser.add_argument('--inner_lr', default=1e-4, type=float, help='learning rate for inner loop')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers in Dataloader') 
    parser.add_argument('--predict_velocity_diff', help='whether to predict change in velocity', action='store_true')
    parser.add_argument('--num_neighbors', default=10, type=int, help='number of neighbors')
    parser.add_argument('--conserve_quantity', default='approx', choices=['approx', 'exact'], type=str, help='conserved quantity')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], type=str, help='cuda or cpu')
    parser.add_argument('--activation', default='ReLU', choices=['ReLU', 'Sigmoid'], type=str, help='activation function')
    parser.add_argument('--activation_noether', default='ReLU', choices=['ReLU', 'Sigmoid'], type=str, help='noether activation function')
    parser.add_argument('--embedding_dim', default=8, type=int, help='dimension of the Noether embedding')
    args = parser.parse_args()

    main(args)
