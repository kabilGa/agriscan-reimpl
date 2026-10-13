"""Near-duplicate detection with CNN embeddings, and a group-aware train/val split."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

_PREP = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class _Images(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return _PREP(Image.open(self.paths[i]).convert("RGB"))


def make_embedder(pretrained: bool = True) -> nn.Module:
    m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier = nn.Identity()  # keep the 1280-d pooled feature, drop the 1000-class head
    return m.eval()


@torch.no_grad()
def embed(paths, model: nn.Module, device: str, batch_size: int = 128, workers: int = 2) -> torch.Tensor:
    model.to(device)
    out = [model(x.to(device)).float().cpu()
           for x in DataLoader(_Images(paths), batch_size=batch_size, num_workers=workers)]
    return torch.cat(out)


def prepare(feats: torch.Tensor, mean: torch.Tensor | None = None):
    """Center, then L2-normalize, so a dot product is a cosine similarity.

    Centering matters: post-ReLU features are all positive, so raw cosines of unrelated
    images already sit around 0.5+, which makes any threshold hard to read.
    """
    feats = feats.float()
    mean = feats.mean(0, keepdim=True) if mean is None else mean
    return F.normalize(feats - mean, dim=1), mean


@torch.no_grad()
def top1(q, ref, device, *, self_match=False, q_labels=None, ref_labels=None,
         other_label=False, chunk=2048):
    """For each row of q: best cosine similarity to a row of ref, and that row's index.

    self_match: q is ref, so hide each row's match with itself.
    other_label: only consider ref rows of a DIFFERENT class (a 'how similar can unrelated
    leaves look' baseline for choosing a threshold).
    """
    ref_d = ref.to(device)
    rl = None if ref_labels is None else torch.as_tensor(np.array(ref_labels)).to(device)  # copy: pandas arrays are read-only
    sims, idxs = [], []
    for s in range(0, len(q), chunk):
        sim = q[s:s + chunk].to(device) @ ref_d.T
        if self_match:
            sim[torch.arange(len(sim)), torch.arange(s, s + len(sim))] = -2
        if other_label:
            ql = torch.as_tensor(np.array(q_labels[s:s + chunk])).to(device)
            sim[ql[:, None] == rl[None, :]] = -2
        v, i = sim.max(1)
        sims.append(v.cpu())
        idxs.append(i.cpu())
    return torch.cat(sims).numpy(), torch.cat(idxs).numpy()


@torch.no_grad()
def pairs_above(feats, thr, device, chunk=2048, max_pairs=50_000_000):
    """All index pairs (i < j) whose cosine similarity is >= thr."""
    f = feats.to(device)
    rows, cols, total = [], [], 0
    for s in range(0, len(f), chunk):
        r, c = (f[s:s + chunk] @ f.T >= thr).nonzero(as_tuple=True)
        r = r + s
        keep = r < c
        rows.append(r[keep].cpu())
        cols.append(c[keep].cpu())
        total += int(keep.sum())
        if total > max_pairs:
            raise RuntimeError(f"More than {max_pairs} pairs at threshold {thr}: threshold too low")
    return torch.cat(rows).numpy(), torch.cat(cols).numpy()


def near_dup_groups(feats, labels, thr, device):
    """Connected components of the 'near-copy' graph; returns (group id per image, #cross-label pairs).

    Only same-label pairs are linked: two near-identical images with different labels would be
    label noise, and linking them would glue unrelated classes into one giant group.
    """
    labels = np.asarray(labels)
    i, j = pairs_above(feats, thr, device)
    same = labels[i] == labels[j]
    n = len(labels)
    graph = coo_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(n, n))
    _, group_id = connected_components(graph, directed=False)
    return group_id, int((~same).sum())


def group_split(labels, groups, val_frac, seed, overshoot=1.25):
    """Per class, send whole groups to validation until ~val_frac of the class is there.

    A group that would push validation past overshoot * target stays in train, so one huge
    group cannot take the whole class (the real fraction is reported by the caller).
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(labels), bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        gids = np.unique(groups[idx])
        rng.shuffle(gids)
        target, got = val_frac * len(idx), 0
        for g in gids:
            if got >= target:
                break
            members = idx[groups[idx] == g]
            if got + len(members) > target * overshoot:
                continue
            is_val[members] = True
            got += len(members)
    return is_val
