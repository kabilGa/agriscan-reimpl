"""Training logic: freezing, schedules, resume, results schema (random init, tiny images, CPU)."""
import json

import torch

from agriscan.leakage import pick_threshold
from agriscan.train import (build_model, default_img_size, ece, epoch_lr, set_stage, set_train_mode, train_run)
from synth import make_dataset


def _steps(model, trainable, n=3):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    for _ in range(n):
        set_train_mode(model, trainable)
        loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 3, 64, 64)), torch.randint(0, 3, (8,)))
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_head_stage_freezes_backbone_including_batchnorm_statistics():
    torch.manual_seed(0)
    model, head, last = build_model("mobilenet_v2", 3, pretrained=False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _steps(model, set_stage(model, head, last, "head"))
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("features."))  # incl. BN running stats
    assert not torch.equal(before["classifier.1.weight"], after["classifier.1.weight"])


def test_ft_stage_changes_only_last_blocks_and_head():
    torch.manual_seed(0)
    model, head, last = build_model("mobilenet_v2", 3, pretrained=False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _steps(model, set_stage(model, head, last, "ft"))
    after = model.state_dict()
    assert torch.equal(before["features.3.conv.0.0.weight"], after["features.3.conv.0.0.weight"])  # early block frozen
    assert not torch.equal(before["features.16.conv.0.0.weight"], after["features.16.conv.0.0.weight"])  # last block trains


def test_lr_schedules():
    assert epoch_lr("constant", 1.0, 3, 4) == 1.0
    assert epoch_lr("cosine", 1.0, 0, 4) == 1.0 and 0 < epoch_lr("cosine", 1.0, 3, 4) < 0.2
    assert [epoch_lr("step", 1.0, e, 6) for e in range(6)] == [1, 1, 1, 1, 0.1, 0.1]


def test_default_sizes_come_from_weights_metadata():
    assert default_img_size("mobilenet_v2") == 224
    assert default_img_size("efficientnet_b3") == 300


def test_ece_perfect_and_overconfident():
    assert ece(torch.tensor([1.0, 1.0]), torch.tensor([True, True])) < 1e-6
    assert ece(torch.tensor([0.95] * 10), torch.tensor([True] * 5 + [False] * 5)) > 0.4


def test_threshold_rule_finds_valley_between_modes():
    rng = torch.Generator().manual_seed(0)
    no_copy = 0.55 + 0.1 * torch.rand(2000, generator=rng)         # val images with no copy in train
    copy = 0.93 + 0.06 * torch.rand(2000, generator=rng)           # val images with a near-copy
    control = 0.30 + 0.25 * torch.rand(5000, generator=rng)        # best match among other classes
    t = pick_threshold(torch.cat([no_copy, copy]).numpy(), control.numpy())
    assert 0.55 <= t["final"] < 0.93


def _cfg(tmp_path, data, **kw):
    cfg = dict(name="t", arch="mobilenet_v2", split="original", root=str(data), csv=None, seed=0, role="core",
               tag="base", epochs_head=1, epochs_ft=2, augment=True, img_size=64, schedule="cosine", lr_head=1e-3,
               lr_ft=1e-4, weight_decay=1e-4, batch=16, workers=0, pretrained=False, save_model=True,
               limit_train=0, limit_val=0, stop_after=0, results_dir=str(tmp_path / "res"),
               ckpt_dir=str(tmp_path / "ck"), model_dir=str(tmp_path / "models"), split_meta=None)
    cfg.update(kw)
    return cfg


def test_train_run_schema_and_resume(tmp_path):
    data = tmp_path / "data"
    make_dataset(data, n_orig=12)
    assert train_run(_cfg(tmp_path, data, stop_after=2)) is None             # "session crashed" after 2 of 3 epochs
    assert (tmp_path / "ck" / "t.pt").exists() and not (tmp_path / "res" / "t.json").exists()
    out = train_run(_cfg(tmp_path, data))                                    # resumes at epoch 3
    m = out["metrics"]
    assert [h["stage"] for h in m["history"]] == ["head", "ft", "ft"]
    assert len(m["confusion_matrix"]) == 3 and set(m["per_class_recall"]) == {"A___healthy", "B___rust", "C___scab"}
    assert {"seed", "config", "date_utc", "env"} <= out.keys() and out["env"]["torch"]
    assert not (tmp_path / "ck" / "t.pt").exists()                           # checkpoint cleaned up
    assert (tmp_path / "models" / "t.pt").exists()
    again = train_run(_cfg(tmp_path, data))                                  # finished runs are skipped
    assert again["date_utc"] == out["date_utc"]


def test_head_only_run_has_no_ft_stage(tmp_path):
    data = tmp_path / "data"
    make_dataset(data, n_orig=6)
    out = train_run(_cfg(tmp_path, data, name="frozen", epochs_head=2, epochs_ft=0, save_model=False))
    assert {h["stage"] for h in out["metrics"]["history"]} == {"head"}
    assert json.loads((tmp_path / "res" / "frozen.json").read_text())["config"]["epochs_ft"] == 0
