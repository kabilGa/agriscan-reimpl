"""Steps 2-4: train ONE configuration (backbone x split x ablation factor) and save its results JSON."""
import argparse
import json
from pathlib import Path

from agriscan.data import find_dataset_root
from agriscan.train import train_run

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["mobilenet_v2", "efficientnet_b3", "resnet50"])
ap.add_argument("--split", required=True, choices=["original", "clean"])
ap.add_argument("--root", default="auto")
ap.add_argument("--csv", default="splits/clean_split.csv.gz")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="base", help="short label of what differs from the baseline (used in the run name)")
ap.add_argument("--role", default="core", choices=["core", "ablation", "seed_replicate", "extra"])
ap.add_argument("--epochs-head", type=int, default=2, help="epochs with the backbone frozen")
ap.add_argument("--epochs-ft", type=int, default=4, help="epochs fine-tuning the last blocks (0 = head only)")
ap.add_argument("--no-aug", action="store_true")
ap.add_argument("--img-size", type=int, default=0, help="0 = size recommended by the weights metadata")
ap.add_argument("--schedule", default="cosine", choices=["cosine", "step", "constant"])
ap.add_argument("--lr-head", type=float, default=1e-3)
ap.add_argument("--lr-ft", type=float, default=1e-4)
ap.add_argument("--weight-decay", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=0, help="0 = per-backbone default")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
ap.add_argument("--save-model", action="store_true")
ap.add_argument("--limit-train", type=int, default=0)
ap.add_argument("--limit-val", type=int, default=0)
ap.add_argument("--stop-after", type=int, default=0, help="test hook: stop after N epochs, as if the session crashed")
ap.add_argument("--results-dir", default="results/runs")
ap.add_argument("--ckpt-dir", default="artifacts/ckpt")
ap.add_argument("--model-dir", default="artifacts/models")
args = ap.parse_args()

root = find_dataset_root() if args.root == "auto" else Path(args.root)
meta = None
if args.split == "clean" and Path("results/step1_clean_split.json").exists():
    meta = json.loads(Path("results/step1_clean_split.json").read_text())["metrics"]["threshold"]
cfg = dict(name=f"{args.arch}__{args.split}__s{args.seed}__{args.tag}", arch=args.arch, split=args.split,
           root=str(root), csv=args.csv, seed=args.seed, role=args.role, tag=args.tag,
           epochs_head=args.epochs_head, epochs_ft=args.epochs_ft, augment=not args.no_aug,
           img_size=args.img_size, schedule=args.schedule, lr_head=args.lr_head, lr_ft=args.lr_ft,
           weight_decay=args.weight_decay, batch=args.batch, workers=args.workers, pretrained=not args.no_pretrained,
           save_model=args.save_model, limit_train=args.limit_train, limit_val=args.limit_val,
           stop_after=args.stop_after, results_dir=args.results_dir, ckpt_dir=args.ckpt_dir, model_dir=args.model_dir,
           split_meta={"clean_threshold": meta} if meta is not None else None)
train_run(cfg)
