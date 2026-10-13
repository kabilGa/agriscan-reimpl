"""Logic tests for the leakage tools on synthetic vectors (no images, no downloads)."""
import numpy as np
import torch

from agriscan.leakage import group_split, near_dup_groups, pairs_above, prepare, top1


def _pool(n_orig=20, copies=3, dim=64, n_classes=3, noise=0.03, seed=0):
    """Each 'original' gets `copies` noisy near-copies, like offline augmentation."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n_classes * n_orig, dim, generator=g)
    x = base.repeat_interleave(copies, 0) + noise * torch.randn(n_classes * n_orig * copies, dim, generator=g)
    y = np.repeat(np.repeat(np.arange(n_classes), n_orig), copies)
    true_group = np.repeat(np.arange(n_classes * n_orig), copies)
    return prepare(x)[0], y, true_group


def test_top1_finds_the_copy_and_hides_self():
    f, _, tg = _pool()
    sim, idx = top1(f, f, "cpu", self_match=True)
    assert (tg[idx] == tg).all()          # nearest other image is a copy of the same original
    assert (sim > 0.9).all()


def test_groups_recover_originals():
    f, y, tg = _pool()
    groups, cross = near_dup_groups(f, y, 0.9, "cpu")
    assert len(np.unique(groups)) == len(np.unique(tg))
    assert cross == 0


def test_naive_split_leaks_but_group_split_does_not():
    f, y, _ = _pool()
    groups, _ = near_dup_groups(f, y, 0.9, "cpu")
    naive = np.random.default_rng(0).random(len(y)) < 0.2        # like splitting augmented files
    clean = group_split(y, groups, 0.2, seed=0)
    for is_val, expect_leak in ((naive, True), (clean, False)):
        sim, _ = top1(f[is_val], f[~is_val], "cpu")
        assert ((sim >= 0.9).sum() > 0) == expect_leak
    assert 0.1 < clean.mean() < 0.35      # still roughly the requested 20%


def test_group_split_keeps_groups_whole_and_is_seeded():
    f, y, _ = _pool()
    groups, _ = near_dup_groups(f, y, 0.9, "cpu")
    a = group_split(y, groups, 0.2, seed=1)
    assert (a == group_split(y, groups, 0.2, seed=1)).all()
    for g in np.unique(groups):
        assert len(set(a[groups == g])) == 1


def test_other_label_baseline_ignores_same_class():
    f, y, _ = _pool()
    sim, idx = top1(f, f, "cpu", q_labels=y, ref_labels=y, other_label=True)
    assert (y[idx] != y).all()
    assert (sim < 0.9).all()              # different originals are not near-copies


def test_pairs_above_counts():
    f, _, _ = _pool(n_orig=5, copies=3, n_classes=1)
    i, j = pairs_above(f, 0.9, "cpu")
    assert len(i) == 5 * 3                # 3 pairs inside each group of 3
    assert (i < j).all()
