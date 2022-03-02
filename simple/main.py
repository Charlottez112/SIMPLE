from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime

from .model import StatePredictor, TemperaturePredictor
from .data import SimulationDataLoader
from .train import train_one_epoch
from .eval import evaluate



def main(args):
    # Initialize models.
    f = StatePredictor(args.num_neighbors)
    g_list: list[nn.Module] = [TemperaturePredictor()]
    
    current_date = datetime.date.today()
    writer = SummaryWriter(f'./Log/{current_date}')

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
        validation_error = evaluate(f, g_list, eval_loader, args)
        print(f"validation error: {validation_error}")
        writer.add_scalar('Val_error', validation_error, epoch)
        writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--outer_lr', default=1e-4, type=float, help='learning rate for outer loop')
    parser.add_argument('--inner_lr', default=1e-4, type=float, help='learning rate for inner loop')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers in Dataloader') 
    parser.add_argument('--num_neighbors', default=10, type=int, help='number of neighbors')
    parser.add_argument('--conserve_quantity', default='approx', type=str, help='conserved quantity')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    
    args = parser.parse_args()

    main(args)
