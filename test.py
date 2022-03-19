# type: ignore
"""Quick and dirty end-to-end testing."""


import torch

from simple.train import train_one_epoch
from simple.eval import evaluate
from simple.model import StatePredictor, TemperaturePredictor, LearnedQuantityPredictor


B = 8
F = 60
N = 256


class Args:
    inner_lr = 1e-4
    conserve_quantity = "approx"


f = StatePredictor(10)
g_list = [LearnedQuantityPredictor(8, N), TemperaturePredictor()]
loader = [
    {"position": torch.rand(B, F, N, 3), "velocity": torch.rand(B, F, N, 3), "boxdim": torch.rand(B, F)},
    {"position": torch.zeros(B, F, N, 3), "velocity": torch.zeros(B, F, N, 3), "boxdim": torch.zeros(B, F)},
    {"position": torch.ones(B, F, N, 3), "velocity": torch.ones(B, F, N, 3), "boxdim": torch.ones(B, F)},
]
outer_params = list(f.parameters())
for g in g_list:
    outer_params.extend(list(g.parameters()))
outer_optimizer = torch.optim.Adam(params=outer_params, lr=1e-4)
args = Args()


print("Train")
train_one_epoch(f, g_list, loader, outer_optimizer, args)

print("Eval")
evaluate(f, g_list, loader, args)