"""Filename-based cross-check of near-duplicate leakage (an EXTRA analysis, independent of embeddings).

Assumption (stated in the results, not verified against the dataset's documentation): in this dataset an
augmented copy keeps the UUID of its source photo, e.g. `<uuid>___RS_HL 1827_flipLR.JPG` is a flipped copy of
`<uuid>___RS_HL 1827.JPG`. Two files with the same UUID are therefore treated as the same source photo.
"""
import re

import pandas as pd

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _name(rel: str) -> str:
    return rel.rsplit("/", 1)[-1]


def source_key(rel: str):
    """The UUID before '___' in the file name, or None if the name does not follow that pattern."""
    head, sep, _ = _name(rel).partition("___")
    return head if sep and UUID.fullmatch(head) else None


def augmentation(rel: str) -> str:
    """Text after the numeric id in the file name ('(none)' = no suffix, e.g. an original), for reporting."""
    stem = re.sub(r"\.(jpg|jpeg|png)$", "", _name(rel), flags=re.I)
    m = re.search(r"\s\d+(.*)$", stem)
    if not m:
        return "(unparsed)"
    return m.group(1).strip() or "(none)"


def original_split(rel: str):
    m = re.search(r"/(train|valid)/", rel)
    return None if not m else ("train" if m.group(1) == "train" else "val")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds key / augmentation / orig columns to a table with columns rel, label, split, group."""
    out = df.copy()
    out["key"] = out.rel.map(source_key)
    out["aug"] = out.rel.map(augmentation)
    out["orig"] = out.rel.map(original_split)
    return out


def sibling_leakage(df: pd.DataFrame, split_col: str) -> dict:
    """How many validation images have a same-source image in the training side of `split_col`."""
    tr = set(df.loc[(df[split_col] == "train") & df.key.notna(), "key"])
    va = df[df[split_col] == "val"]
    keyed = va.key.notna()
    leak = va.key.isin(tr) & keyed
    by_aug = (va.assign(leak=leak).groupby("aug").leak.agg(["size", "sum"]).sort_values("size", ascending=False))
    return {"n_val": int(len(va)), "n_val_with_key": int(keyed.sum()), "with_sibling_in_train": int(leak.sum()),
            "fraction_of_all_val": float(leak.sum() / len(va)),
            "fraction_of_keyed_val": float(leak.sum() / max(int(keyed.sum()), 1)),
            "by_augmentation": {a: {"n_val": int(r["size"]), "with_sibling_in_train": int(r["sum"])} for a, r in by_aug.iterrows()}}


def group_capture(df: pd.DataFrame) -> dict:
    """How well the embedding-based groups (column `group`) kept same-source images together."""
    k = df[df.key.notna()].copy()
    k["n_same_key"] = k.groupby("key").key.transform("size")
    multi = k[k.n_same_key > 1].copy()
    multi["n_same_key_and_group"] = multi.groupby(["key", "group"]).key.transform("size")
    multi["captured"] = multi.n_same_key_and_group > 1          # shares its group with at least one same-source image
    keys_whole = (multi.groupby("key").group.nunique() == 1)
    by_aug = multi.groupby("aug").captured.agg(["size", "sum"]).sort_values("size", ascending=False)
    return {"n_keyed_images": int(len(k)), "n_sources": int(k.key.nunique()),
            "images_per_source_mean": float(k.groupby("key").size().mean()), "images_per_source_max": int(k.groupby("key").size().max()),
            "sources_with_more_than_one_image": int(len(keys_whole)),
            "sources_kept_in_one_group": int(keys_whole.sum()),
            "images_with_a_sibling": int(len(multi)), "images_grouped_with_a_sibling": int(multi.captured.sum()),
            "sources_spanning_multiple_labels": int((k.groupby("key").label.nunique() > 1).sum()),
            "captured_by_augmentation": {a: {"n_images_with_sibling": int(r["size"]), "grouped_with_a_sibling": int(r["sum"])}
                                         for a, r in by_aug.iterrows()}}
