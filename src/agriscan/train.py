"""Transfer learning for the leaf classifiers: train a new head first, then fine-tune the last blocks."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import (EfficientNet_B3_Weights, MobileNet_V2_Weights, ResNet50_Weights,
                                efficientnet_b3, mobilenet_v2, resnet50)

from agriscan.data import find_splits, list_images
from agriscan.utils import save_results, set_seed

ARCHS = {"mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1),
         "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
         "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1)}
DEFAULT_BATCH = {"mobilenet_v2": 64, "efficientnet_b3": 32, "resnet50": 64}  # B3 at 300px is memory hungry
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def default_img_size(arch: str) -> int:
    """Input size recommended by the torchvision weights' own metadata (B3: 300, others: 224)."""
    return int(ARCHS[arch][1].transforms().crop_size[0])


def build_model(arch: str, num_classes: int, pretrained: bool = True):
    """Returns (model, head, last_blocks). `last_blocks` are what stage 2 unfreezes."""
    fn, weights = ARCHS[arch]
    m = fn(weights=weights if pretrained else None)
    if arch == "resnet50":
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, m.fc, [m.layer4]
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)  # keeps torchvision's dropout
    start = 13 if arch == "mobilenet_v2" else 6   # last 5 inverted-residual blocks+conv / last 3 stages
    return m, m.classifier, list(m.features[start:])


def set_stage(model, head, last, stage: str):
    """stage 'head': only the new head learns. stage 'ft': head + last blocks learn."""
    for p in model.parameters():
        p.requires_grad = False
    trainable = [head] + (last if stage == "ft" else [])
    for mod in trainable:
        for p in mod.parameters():
            p.requires_grad = True
    return trainable


def set_train_mode(model, trainable):
    # Frozen BatchNorm layers must stay in eval mode, otherwise their running statistics would
    # keep drifting even though their weights are frozen.
    model.eval()
    for mod in trainable:
        mod.train()


def epoch_lr(schedule: str, base: float, e: int, n: int) -> float:
    if schedule == "cosine":
        return base * 0.5 * (1 + math.cos(math.pi * e / n))
    if schedule == "step":
        return base * (0.1 if e >= math.ceil(2 * n / 3) else 1.0)
    return base  # constant


def make_transforms(size: int, augment: bool):
    norm = T.Normalize(MEAN, STD)
    ev = T.Compose([T.Resize((size, size)), T.ToTensor(), norm])
    if not augment:
        return ev, ev
    tr = T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                    T.RandomRotation(20), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    return tr, ev


class ImageDataset(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths, self.labels, self.tfm = list(paths), list(labels), tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tfm(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


CSV_SPLITS = ("clean", "photo", "leaf")  # clean: embedding groups (train/val); photo, leaf: grouped (train/val/test)


def _read_split_csv(csv, root):
    c = pd.read_csv(csv)
    c["path"] = c.rel.map(lambda r: str(Path(root) / r))
    return c


def load_split(split: str, root, csv=None):
    """'original' = the dataset's own train/valid folders; 'clean', 'photo', 'leaf' = our group-aware CSV files."""
    root = Path(root)
    if split == "original":
        d = find_splits(root)
        tr, va = list_images(d["train"], "train", root), list_images(d["valid"], "valid", root)
        classes = sorted(set(tr.label) | set(va.label))
    elif split in CSV_SPLITS:
        c = _read_split_csv(csv, root)
        tr, va = c[c.split == "train"], c[c.split == "val"]
        classes = sorted(set(c.label))  # includes test-only labels, if any
    else:
        raise ValueError(split)
    return tr.reset_index(drop=True), va.reset_index(drop=True), classes


def load_test(split: str, root, csv=None):
    """The held-out test rows of a CSV split, or None if the split has none (original, clean)."""
    if split not in CSV_SPLITS:
        return None
    c = _read_split_csv(csv, root)
    t = c[c.split == "test"]
    return t.reset_index(drop=True) if len(t) else None


def report(stats: dict, df: pd.DataFrame, cidx: dict, classes: list) -> dict:
    """Metrics of one evaluation pass, plus the same accuracy on files covered / not covered by the authors' leaf map."""
    nc = len(classes)
    hit = stats["hit"].numpy().astype(bool)
    y = df.label.map(cidx).to_numpy()
    mapped = df["leaf_mapped"].to_numpy(bool) if "leaf_mapped" in df else np.zeros(len(df), bool)

    def macro(mask):
        r = [hit[mask & (y == c)].mean() for c in range(nc) if (mask & (y == c)).any()]
        return float(np.mean(r)) if r else None

    def part(mask):
        return {"n": int(mask.sum()), "acc": float(hit[mask].mean()) if mask.any() else None, "macro_recall": macro(mask)}

    cm = stats["cm"]
    support = cm.sum(1)
    return {"n": int(len(df)), "acc": stats["acc"], "loss": stats["loss"], "macro_recall": macro(np.ones(len(df), bool)),
            "per_class_recall": dict(zip(classes, [float(cm[i, i] / support[i]) if support[i] > 0 else None for i in range(nc)])),
            "confusion_matrix": cm.tolist(), "ece_15bins": ece(stats["conf"], stats["hit"]),
            "leaf_mapped": {**part(mapped), "fraction": float(mapped.mean())}, "not_leaf_mapped": part(~mapped)}


def ece(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error: gap between confidence and accuracy, averaged over confidence bins."""
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(correct[m].float().mean().item() - conf[m].mean().item())
    return total


def run_epoch(model, loader, device, loss_fn, opt=None, scaler=None, amp=False, nc=0):
    """One pass over loader. Trains if opt is given, otherwise evaluates and returns extra stats."""
    training = opt is not None
    loss_sum = correct = n = 0
    cm = torch.zeros(nc * nc, dtype=torch.long) if not training else None
    confs, hits = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = loss_fn(out, y)
            if training:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            pred = out.argmax(1)
            loss_sum += loss.item() * len(y)
            correct += (pred == y).sum().item()
            n += len(y)
            if not training:
                cm += torch.bincount((y * nc + pred).cpu(), minlength=nc * nc)
                conf = out.float().softmax(1).max(1).values
                confs.append(conf.cpu())
                hits.append((pred == y).cpu())
    stats = {"loss": loss_sum / n, "acc": correct / n}
    if not training:
        stats.update(cm=cm.view(nc, nc), conf=torch.cat(confs), hit=torch.cat(hits))
    return stats


def measure_speed(model, loader, device, amp, iters=20):
    """Forward-only images/second on one batch (no data loading), to compare backbones."""
    x, _ = next(iter(loader))
    x = x.to(device)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
        for _ in range(3):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return len(x) * iters / (time.time() - t)


def save_checkpoint(path, model, arch, classes, img_size, extra=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "classes": classes, "img_size": img_size,
                "state_dict": model.state_dict(), "extra": extra or {}}, path)


def train_run(cfg: dict):
    """Train one configuration end to end. Returns the results dict, or None if stopped early (test hook)."""
    name = cfg["name"]
    res_path = Path(cfg["results_dir"]) / f"{name}.json"
    if res_path.exists():
        print(f"[{name}] results already exist, skipping")
        return json.loads(res_path.read_text())

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    tr, va, classes = load_split(cfg["split"], cfg["root"], cfg.get("csv"))
    if cfg.get("limit_train"):
        tr = tr.sample(min(cfg["limit_train"], len(tr)), random_state=cfg["seed"]).reset_index(drop=True)
    if cfg.get("limit_val"):
        va = va.sample(min(cfg["limit_val"], len(va)), random_state=cfg["seed"]).reset_index(drop=True)
    te = load_test(cfg["split"], cfg["root"], cfg.get("csv"))
    if te is not None and cfg.get("limit_test"):
        te = te.sample(min(cfg["limit_test"], len(te)), random_state=cfg["seed"]).reset_index(drop=True)
    cidx = {c: i for i, c in enumerate(classes)}
    nc = len(classes)

    size = cfg["img_size"] or default_img_size(cfg["arch"])
    batch = cfg["batch"] or DEFAULT_BATCH[cfg["arch"]]
    tfm_tr, tfm_ev = make_transforms(size, cfg["augment"])
    w = cfg["workers"]
    kw = dict(batch_size=batch, num_workers=w, pin_memory=amp, persistent_workers=w > 0)
    g = torch.Generator().manual_seed(cfg["seed"])
    dl_tr = DataLoader(ImageDataset(tr.path, tr.label.map(cidx), tfm_tr), shuffle=True, generator=g, **kw)
    dl_va = DataLoader(ImageDataset(va.path, va.label.map(cidx), tfm_ev), shuffle=False, **kw)

    model, head, last = build_model(cfg["arch"], nc, cfg["pretrained"])
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    plan = [("head", e, cfg["epochs_head"]) for e in range(cfg["epochs_head"])] + \
           [("ft", e, cfg["epochs_ft"]) for e in range(cfg["epochs_ft"])]

    ck_path = Path(cfg["ckpt_dir"]) / f"{name}.pt"
    history, start, resumed, opt_state, opt_stage, t_prev = [], 0, False, None, None, 0.0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck["model"])
        history, start, opt_state, opt_stage, t_prev = ck["history"], ck["next"], ck["opt"], ck["opt_stage"], ck["secs"]
        resumed = True
        print(f"[{name}] resuming at epoch {start}/{len(plan)}")

    t0, cur, opt, trainable = time.time(), None, None, None
    for i in range(start, len(plan)):
        stage, e, n = plan[i]
        if stage != cur:  # entering a new stage: new frozen/unfrozen set and a fresh optimizer
            cur = stage
            trainable = set_stage(model, head, last, stage)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=cfg["lr_head"] if stage == "head" else cfg["lr_ft"],
                                    weight_decay=cfg["weight_decay"])
            if resumed and opt_stage == stage:
                opt.load_state_dict(opt_state)
            resumed = False
        base = cfg["lr_head"] if stage == "head" else cfg["lr_ft"]
        lr = base if stage == "head" else epoch_lr(cfg["schedule"], base, e, n)  # head stage: constant lr
        for gp in opt.param_groups:
            gp["lr"] = lr
        t_ep = time.time()
        set_train_mode(model, trainable)
        trn = run_epoch(model, dl_tr, device, loss_fn, opt, scaler, amp)
        model.eval()
        val = run_epoch(model, dl_va, device, loss_fn, amp=amp, nc=nc)
        rec = {"stage": stage, "epoch": e + 1, "lr": lr, "train_loss": trn["loss"], "train_acc": trn["acc"],
               "val_loss": val["loss"], "val_acc": val["acc"], "seconds": time.time() - t_ep}
        history.append(rec)
        print(f"[{name}] {stage} {e + 1}/{n} lr={lr:.2e} train {trn['loss']:.4f}/{trn['acc']:.4f} "
              f"val {val['loss']:.4f}/{val['acc']:.4f}  {rec['seconds']:.0f}s", flush=True)
        Path(ck_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "opt_stage": cur, "history": history,
                    "next": i + 1, "secs": t_prev + time.time() - t0}, str(ck_path) + ".tmp")
        Path(str(ck_path) + ".tmp").replace(ck_path)  # atomic: a crash never leaves half a checkpoint
        if cfg.get("stop_after") and i + 1 >= cfg["stop_after"]:
            return None

    cm = val["cm"]
    support = cm.sum(1)
    recall = [float(cm[i, i] / support[i]) if support[i] > 0 else None for i in range(nc)]
    n_params = sum(p.numel() for p in model.parameters())
    metrics = {"val_acc": val["acc"], "val_loss": val["loss"],
               "macro_recall": float(np.mean([r for r in recall if r is not None])),
               "per_class_recall": dict(zip(classes, recall)), "per_class_support": dict(zip(classes, support.tolist())),
               "confusion_matrix": cm.tolist(), "classes": classes,
               "ece_15bins": ece(val["conf"], val["hit"]), "mean_confidence": float(val["conf"].mean()),
               "history": history, "train_seconds": t_prev + time.time() - t0,
               "n_params": n_params, "model_size_mb_fp32": n_params * 4 / 1e6,
               "forward_img_per_s": measure_speed(model, dl_va, device, amp), "speed_device": device,
               "n_train": len(tr), "n_val": len(va), "img_size": size, "batch": batch,
               "final_epoch_reported": "last epoch (no best-epoch selection on validation)"}
    if "leaf_mapped" in va:  # validation accuracy split by whether the authors' leaf map covers the file
        vr = report(val, va, cidx, classes)
        metrics["val_leaf_mapped"], metrics["val_not_leaf_mapped"] = vr["leaf_mapped"], vr["not_leaf_mapped"]
    if te is not None:
        # The held-out test split is evaluated ONCE, here, after the last epoch. It is not in `history` and nothing was
        # selected or tuned on it.
        dl_te = DataLoader(ImageDataset(te.path, te.label.map(cidx), tfm_ev), shuffle=False, **kw)
        model.eval()
        metrics["test"] = report(run_epoch(model, dl_te, device, loss_fn, amp=amp, nc=nc), te, cidx, classes)
        metrics["test"]["evaluated_once_after_final_epoch"] = True
        print(f"[{name}] test (once): acc={metrics['test']['acc']:.4f} | leaf-mapped subset acc={metrics['test']['leaf_mapped']['acc']} "
              f"(n={metrics['test']['leaf_mapped']['n']} of {metrics['test']['n']})")
    if cfg.get("split_meta"):
        metrics["split_meta"] = cfg["split_meta"]
    save_results(res_path, name=name, seed=cfg["seed"], config=cfg, metrics=metrics)
    if cfg.get("save_model"):
        save_checkpoint(Path(cfg["model_dir"]) / f"{name}.pt", model, cfg["arch"], classes, size,
                        {"val_acc": val["acc"], "split": cfg["split"]})
    ck_path.unlink(missing_ok=True)
    print(f"[{name}] done: val_acc={val['acc']:.4f} macro_recall={metrics['macro_recall']:.4f} "
          f"total {metrics['train_seconds'] / 60:.1f} min")
    return json.loads(res_path.read_text())
