"""Photo/leaf grouping and the 70/15/15 split, on synthetic PlantVillage-like names (no dataset, no network)."""
import numpy as np
import pandas as pd

from agriscan.data import find_splits, list_images
from agriscan.grouping import (annotate, build_grouped_splits, files_hash, names, split_checks, split_hash,
                               three_way_split, union_groups)
from synth import make_grouped_dataset


def _files(tmp_path, **kw):
    root = tmp_path / "data"
    leaf_map = make_grouped_dataset(root, **kw)
    dirs = find_splits(root)
    f = pd.concat([list_images(dirs[s], s, root) for s in ("train", "valid")])[["rel", "label"]]
    return f, leaf_map


def test_names_strip_extension_augmentation_and_copy():
    assert names("x/a/11111111-2222-3333-4444-555555555555___RS_HL 1827_flipLR.JPG") == ("rs_hl 1827", "rs_hl 1827")
    assert names("x/a/u___RS_HL 9874 copy_new30degFlipLR.JPG") == ("rs_hl 9874 copy", "rs_hl 9874")
    assert names("x/a/RS_Rust 1563_180deg.JPG")[0] == "rs_rust 1563"        # no uuid part (the Corn class)
    assert names("x/a/u___RS_HL 5 copy 2_flipTB.JPG")[1] == "rs_hl 5"


def test_union_groups_links_copies_missing_uuids_and_is_canonical(tmp_path):
    files, leaf_map = _files(tmp_path)
    ann = annotate(files, leaf_map)
    g = union_groups(ann, ["uuid", "class_base"])
    ann["g"] = g
    # a ' copy' file has its own uuid but must share a group with its base photo
    copy = ann[ann.rel.str.contains(" copy")].iloc[0]
    base_rows = ann[(ann.class_base == copy.class_base) & ~ann.rel.str.contains(" copy")]
    assert len(base_rows) and (base_rows.g == copy.g).all()
    # class C has no uuid at all: its augmentations still end up together via the base name
    c = ann[ann.label == "C___scab"]
    assert c.uuid.isna().all() and c.groupby("class_base").g.nunique().max() == 1
    # canonical ids: shuffling the row order does not change which files group together
    shuffled = annotate(files.sample(frac=1, random_state=1), leaf_map)
    g2 = pd.Series(union_groups(shuffled, ["uuid", "class_base"]), index=shuffled.rel)
    assert (g2.loc[ann.rel].to_numpy() == g).all()


def test_leaf_id_respects_the_class_prefix(tmp_path):
    files, leaf_map = _files(tmp_path)
    ann = annotate(files, leaf_map)
    row = ann[(ann.label == "A___healthy") & (ann.stem == "rs_hl 100")].iloc[0]
    assert row.leaf.startswith("A___healthy:::")          # the 'Z___other' id for the same stem is ignored
    assert 0.5 < ann.leaf_mapped.mean() < 0.95            # some sources are not covered by the map


def test_splits_are_whole_group_balanced_and_deterministic(tmp_path):
    files, leaf_map = _files(tmp_path, n_sources=60)
    ann, sp = build_grouped_splits(files, leaf_map, seed=0)
    for name, df in sp.items():
        chk = split_checks(ann, df)
        assert chk["groups_straddling_parts"] == 0 and chk["uuids_straddling_parts"] == 0
        assert chk["class_base_names_straddling_parts"] == 0
        n = chk["n"]
        assert abs(n["val"] / sum(n.values()) - 0.15) < 0.08 and abs(n["test"] / sum(n.values()) - 0.15) < 0.08
    assert split_checks(ann, sp["leaf"])["leaf_ids_straddling_parts"] == 0
    # photo-only grouping leaves shared leaves across parts; leaf grouping removes them on mapped files
    assert split_checks(ann, sp["photo"])["leaf_ids_straddling_parts"] > 0
    assert split_hash(sp["leaf"]) == split_hash(build_grouped_splits(files, leaf_map, seed=0)[1]["leaf"])      # same seed
    assert split_hash(sp["leaf"]) != split_hash(build_grouped_splits(files, leaf_map, seed=1)[1]["leaf"])      # other seed
    assert files_hash(files) == files_hash(files.sample(frac=1, random_state=3))                                  # order-independent


def test_three_way_split_respects_class_quotas():
    labels = np.repeat(["a", "b"], 100)
    groups = np.arange(200) // 2                     # groups of 2
    out = three_way_split(labels, groups, seed=0)
    for lab in ("a", "b"):
        part = out[labels == lab]
        assert 10 <= (part == "val").sum() <= 20 and 10 <= (part == "test").sum() <= 20
    for g in np.unique(groups):
        assert len(set(out[groups == g])) == 1


def test_documented_class_and_code_aliases():
    from agriscan.grouping import leaf_id
    m = {"rs_hl 5": ["Apple_Frogeye Spot:::3.0", "Soybean___healthy:::9.0"], "com.g_fl 0004": ["Tomato___Target_Spot:::2.0", "Tomato___Early_blight:::7.0"]}
    assert leaf_id("rs_hl 5", "rs_hl 5", "Apple___Black_rot", m) == "Apple_Frogeye Spot:::3.0"          # class alias; other crop ignored
    assert leaf_id("rs_hl 5", "rs_hl 5", "Apple___healthy", m) is None                                    # no alias for other classes
    assert leaf_id("com.g_tgs_fl 0004", "com.g_tgs_fl 0004", "Tomato___Target_Spot", m) == "Tomato___Target_Spot:::2.0"   # code alias
    assert leaf_id("com.g_tgs_fl 0004", "com.g_tgs_fl 0004", "Tomato___Early_blight", m) is None
