"""Source-photo and leaf grouping for leakage-free splits of the PlantVillage-derived dataset.

Grouping keys (all derived from file names, plus the original authors' leaf map):
  photo group = images linked by the same UUID prefix OR the same (class, base file name), where the base name has the
                extension, the augmentation suffix and ' copy N' removed. This handles three facts seen in the original
                repository: augmented copies keep the UUID; a ' copy' file has its OWN uuid; one whole class has no uuid.
  leaf group  = photo groups additionally linked by the authors' leaf id (leaf_grouping/leaf-map.json), where one exists.
The leaf map is downloaded at run time and never committed: the original repository states no licence.
"""
import gzip
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from agriscan.crosscheck import source_key

LEAF_MAP_URL = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/leaf_grouping/leaf-map.json"
FRACS = (0.70, 0.15, 0.15)  # train / val / test
_AUG = re.compile(r"(_flipLR|_flipTB|_\d+deg|_new\w+)+$")
_COPY = re.compile(r"\s*copy\s*\d*$")


def names(rel: str):
    """(stem, base): file name after '___' without extension and augmentation suffix; base also drops ' copy N'."""
    n = re.sub(r"\.(jpg|jpeg|png)$", "", rel.rsplit("/", 1)[-1], flags=re.I)
    n = n.partition("___")[2] or n
    stem = _AUG.sub("", n).lower().strip()
    return stem, _COPY.sub("", stem).strip()


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_leaf_map(path, url: str = LEAF_MAP_URL, retries: int = 3):
    """Read the authors' leaf map from `path`, downloading it first if absent. Returns (map, sha256, downloaded)."""
    path = Path(path)
    downloaded = False
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "agriscan-reimpl"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = r.read()
                    raw = gzip.decompress(raw) if r.headers.get("Content-Encoding") == "gzip" else raw
                json.loads(raw)  # a truncated download fails here and is retried
                path.write_bytes(raw)
                downloaded = True
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise RuntimeError(f"could not download the leaf map from {url}: {e}") from e
                time.sleep(2)
    return json.loads(path.read_text(encoding="utf-8")), sha256_file(path), downloaded


def leaf_id(stem, base, label, leaf_map):
    """The authors' leaf id for a photo, restricted to the photo's own class (stems repeat across classes)."""
    for k in (stem, base):
        c = [leaf for leaf in leaf_map.get(k, []) if leaf.split(":::")[0] == label]
        if len(c) == 1:
            return c[0]
    return None


def annotate(files: pd.DataFrame, leaf_map: dict | None = None) -> pd.DataFrame:
    """files: columns rel, label. Adds uuid, stem, base, class_base and leaf (None if unknown)."""
    d = files.sort_values("rel").reset_index(drop=True)[["rel", "label"]].copy()
    d["uuid"] = d.rel.map(source_key)
    s = d.rel.map(names)
    d["stem"], d["base"] = s.str[0], s.str[1]
    d["class_base"] = d.label + "|" + d.base
    d["leaf"] = [leaf_id(a, b, c, leaf_map) for a, b, c in zip(d.stem, d.base, d.label)] if leaf_map else None
    d["leaf_mapped"] = d.leaf.notna() if leaf_map else False
    return d


def union_groups(d: pd.DataFrame, columns) -> np.ndarray:
    """Group ids of images linked by ANY shared value in `columns` (NaN never links). Ids are canonical: they depend
    only on the file names, not on the graph library's labelling."""
    n, offset, rows, cols = len(d), 0, [], []
    for col in columns:
        codes, _ = pd.factorize(d[col])
        ok = codes >= 0
        if ok.any():
            rows.append(np.flatnonzero(ok))
            cols.append(n + offset + codes[ok])
            offset += int(codes.max()) + 1
    if not rows:
        return np.arange(n)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n + offset, n + offset))
    _, comp = connected_components(graph, directed=False)
    first = pd.Series(d.rel.to_numpy()).groupby(comp[:n]).transform("min")
    return pd.factorize(first, sort=True)[0]


def three_way_split(labels, groups, fracs=FRACS, seed=0, overshoot=1.3) -> np.ndarray:
    """Per class, give whole groups to test, then validation, until ~fracs of the class is there; the rest is train.
    A group that would push a part past `overshoot` times its target goes elsewhere (usually to train)."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    rng = np.random.default_rng(seed)
    out = np.full(len(labels), "train", dtype=object)
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        gids = np.unique(groups[idx])
        rng.shuffle(gids)
        target = {"test": fracs[2] * len(idx), "val": fracs[1] * len(idx)}
        got = {"test": 0, "val": 0}
        for g in gids:
            members = idx[groups[idx] == g]
            for part in ("test", "val"):
                if got[part] < target[part] and got[part] + len(members) <= target[part] * overshoot:
                    out[members] = part
                    got[part] += len(members)
                    break
    return out


def build_grouped_splits(files: pd.DataFrame, leaf_map: dict, seed: int = 0):
    """Returns (annotated files, {'photo': df, 'leaf': df}); each split df has rel, label, split, group, leaf_mapped."""
    ann = annotate(files, leaf_map)
    out = {}
    for name, cols in (("photo", ["uuid", "class_base"]), ("leaf", ["uuid", "class_base", "leaf"])):
        g = union_groups(ann, cols)
        out[name] = pd.DataFrame({"rel": ann.rel, "label": ann.label, "split": three_way_split(ann.label, g, seed=seed),
                                  "group": g, "leaf_mapped": ann.leaf_mapped})
    return ann, out


def split_hash(df: pd.DataFrame) -> str:
    """Order-independent fingerprint of an assignment (rel, label, split)."""
    rows = sorted(f"{r}\t{lab}\t{s}" for r, lab, s in zip(df.rel, df.label, df.split))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def files_hash(files: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(f"{r}\t{lab}" for r, lab in zip(files.rel, files.label))).encode("utf-8")).hexdigest()


def split_checks(ann: pd.DataFrame, sp: pd.DataFrame) -> dict:
    """Measured properties of one split (nothing here is assumed)."""
    d = ann.assign(split=sp.split.to_numpy(), group=sp.group.to_numpy())
    share = d.groupby("label").split.value_counts(normalize=True).unstack().fillna(0)
    sizes = d.group.value_counts()
    tr_leaf = set(d.loc[(d.split == "train") & d.leaf.notna(), "leaf"])
    leak = {}
    for part in ("val", "test"):
        m = d[(d.split == part) & d.leaf.notna()]
        leak[part] = {"n_with_leaf_id": int(len(m)), "share_whose_leaf_is_in_train": float(m.leaf.isin(tr_leaf).mean()) if len(m) else None}
    straddle = lambda col: int((d[d[col].notna()].groupby(col).split.nunique() > 1).sum())  # noqa: E731
    return {"n": {p: int((d.split == p).sum()) for p in ("train", "val", "test")},
            "per_class_share_min_max": {p: [float(share[p].min()), float(share[p].max())] for p in ("train", "val", "test")},
            "n_groups": int(len(sizes)), "largest_group": int(sizes.max()), "groups_over_100_images": int((sizes > 100).sum()),
            "groups_straddling_parts": int((d.groupby("group").split.nunique() > 1).sum()),
            "uuids_straddling_parts": straddle("uuid"), "class_base_names_straddling_parts": straddle("class_base"),
            "leaf_ids_straddling_parts": straddle("leaf"), "same_leaf_leakage_of_mapped_files": leak,
            "test_files_with_leaf_id": int(((d.split == "test") & d.leaf_mapped).sum())}
