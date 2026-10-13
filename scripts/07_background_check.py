"""OPTIONAL: background-only check (inspired by Noyan 2022, arXiv 2206.04374).

Train a tiny MLP on nothing but the colour statistics of the four 16x16 image corners and see how far
above chance it gets. If it is far above chance, background (not leaf) carries label information,
which makes the dataset's accuracy numbers easier than a real field task.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from agriscan.data import find_dataset_root
from agriscan.train import load_split
from agriscan.utils import save_results, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="auto")
ap.add_argument("--csv", default="splits/clean_split.csv.gz")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--patch", type=int, default=16)
ap.add_argument("--results", default="results/extra_background_check.json")
args = ap.parse_args()


class Corners(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = np.asarray(Image.open(self.paths[i]).convert("RGB"), dtype=np.float32) / 255
        p, h, w = args.patch, a.shape[0], a.shape[1]
        patches = [a[:p, :p], a[:p, w - p:], a[h - p:, :p], a[h - p:, w - p:]]
        return torch.tensor(np.concatenate([np.r_[x.mean((0, 1)), x.std((0, 1))] for x in patches]))  # 24 numbers


def features(paths):
    return torch.cat([b for b in DataLoader(Corners(paths), batch_size=256, num_workers=args.workers)])


def fit_eval(xtr, ytr, xva, yva, nc, seed):
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    net = nn.Sequential(nn.Linear(xtr.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, nc))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(30):
        for idx in torch.randperm(len(xtr)).split(512):
            opt.zero_grad()
            nn.functional.cross_entropy(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        return float((net(xva).argmax(1) == yva).float().mean())


set_seed(args.seed)
root = find_dataset_root() if args.root == "auto" else Path(args.root)
metrics = {}
for split in ("original", "clean"):
    tr, va, classes = load_split(split, root, args.csv)
    if args.limit:
        tr, va = tr.sample(min(args.limit, len(tr)), random_state=0), va.sample(min(args.limit, len(va)), random_state=0)
    cidx = {c: i for i, c in enumerate(classes)}
    ytr, yva = torch.tensor(tr.label.map(cidx).to_numpy()), torch.tensor(va.label.map(cidx).to_numpy())
    acc = fit_eval(features(tr.path), ytr, features(va.path), yva, len(classes), args.seed)
    metrics[split] = {"corner_only_val_acc": acc, "chance_uniform": 1 / len(classes),
                      "majority_class_val_acc": float(pd.Series(va.label).value_counts(normalize=True).max()),
                      "n_train": len(tr), "n_val": len(va)}
    print(split, metrics[split])
save_results(args.results, name="extra_background_check", seed=args.seed,
             config={"patch": args.patch, "features": "mean+std RGB of 4 corner patches (24 numbers)", "mlp": "24-256-256-C, 30 epochs Adam"},
             metrics={**metrics, "label": "EXTRA (optional experiment)"})
