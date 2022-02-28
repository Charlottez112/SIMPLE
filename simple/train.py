import argparse

import torch
import torch.optim as optim

from model import StatePredictor, TemperaturePredictor
from loss import task_error, noether_loss_approx
from data import SimulationDataLoader


def main(args):
    f = StatePredictor(args.len_init, args.num_neighbors)
    g = [TemperaturePredictor()]

    train_loader = SimulationDataLoader(split="train", args)
    eval_loader = SimulationDataLoader(split="eval", args)

    # Training loop.
    for i, data in enumerate(train_loader):
        position = data["position"]
        velocity = data["velocity"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
