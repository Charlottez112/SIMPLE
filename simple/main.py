from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .model import StatePredictor, TemperaturePredictor
from .data import SimulationDataLoader
from .train import train_one_epoch
from .eval import evaluate


def main(args):
    # Initialize models.
    f = StatePredictor(args.num_neighbors)
    g_list: list[nn.Module] = [TemperaturePredictor()]

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
        train_one_epoch(f, g_list, train_loader, outer_optimizer, args)
        validation_error = evaluate(f, g_list, eval_loader, args)
        print(f"validation error: {validation_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
