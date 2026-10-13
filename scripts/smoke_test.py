"""Smoke test (CPU, or GPU if present): proves the install works and one training step runs.

It uses random tensors and random weights, so it says NOTHING about accuracy.
Its only job is to catch broken environments before a long Kaggle run.
"""
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from agriscan.utils import save_results, set_seed

SEED = 0
NUM_CLASSES = 3

set_seed(SEED)
# weights=None -> no download, so this also runs offline.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = mobilenet_v2(weights=None, num_classes=NUM_CLASSES).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

x = torch.randn(4, 3, 224, 224, device=device)
y = torch.randint(0, NUM_CLASSES, (4,), device=device)

model.train()
loss = loss_fn(model(x), y)
opt.zero_grad()
loss.backward()
opt.step()

assert torch.isfinite(loss), "loss is not finite"
out = save_results("artifacts/smoke_test.json", name="smoke_test", seed=SEED,
                   config={"model": "mobilenet_v2", "weights": None, "batch": 4},
                   metrics={"one_step_loss": loss.item()})
print(f"OK: one training step ran, loss={loss.item():.4f}, wrote {out}")
