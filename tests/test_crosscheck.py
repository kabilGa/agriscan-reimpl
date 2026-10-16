"""Filename cross-check logic on a small synthetic table (no dataset needed)."""
import pandas as pd

from agriscan.crosscheck import augmentation, group_capture, original_split, prepare, sibling_leakage, source_key

U1, U2, U3 = ("11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222",
              "33333333-3333-3333-3333-333333333333")


def rel(split, uuid, tail):
    return f"x/{split}/Class_A/{uuid}___RS_HL {tail}.JPG"


def test_key_suffix_and_original_split():
    r = rel("valid", U1, "1827_flipLR")
    assert source_key(r) == U1 and augmentation(r) == "_flipLR" and original_split(r) == "val"
    assert augmentation(rel("train", U1, "1827")) == "(none)"
    assert augmentation(rel("train", U1, "1827 copy 2_flipLR")) == "copy 2_flipLR"
    assert source_key("x/train/Class_A/Apple_scab 12.JPG") is None


def test_sibling_leakage_and_group_capture():
    rows = [  # (rel, label, split, group): U1 siblings straddle the split and were put in different groups
        (rel("train", U1, "1"), "A", "train", 0), (rel("valid", U1, "1_flipLR"), "A", "val", 1),
        (rel("train", U2, "2"), "A", "train", 2), (rel("train", U2, "2_90deg"), "A", "train", 2),   # kept together
        (rel("valid", U3, "3"), "A", "val", 3),                                                      # no sibling anywhere
        ("x/valid/Class_A/Apple_scab 12.JPG", "A", "val", 4),                                        # no parsable id
    ]
    df = prepare(pd.DataFrame(rows, columns=["rel", "label", "split", "group"]))
    s = sibling_leakage(df, "split")
    assert s["n_val"] == 3 and s["n_val_with_key"] == 2 and s["with_sibling_in_train"] == 1
    assert s["fraction_of_all_val"] == 1 / 3 and s["fraction_of_keyed_val"] == 1 / 2
    g = group_capture(df)
    assert g["sources_with_more_than_one_image"] == 2 and g["sources_kept_in_one_group"] == 1
    assert g["images_with_a_sibling"] == 4 and g["images_grouped_with_a_sibling"] == 2


def test_original_repo_naming_facts():
    from agriscan.crosscheck import original_repo_naming
    paths = [f"Cls_A/{U1}___RS_HL 1.JPG", f"Cls_A/{U2}___RS_HL 1 copy.JPG",   # copy with its own uuid
             f"Cls_A/{U3}___RS_HL 2.JPG", "Cls_B/RS_Rust 7.JPG", "Cls_B/RS_Rust 8.JPG"]   # class without uuids
    r = original_repo_naming(paths)
    assert r["n_files"] == 5 and r["n_classes"] == 2 and r["n_with_uuid_prefix"] == 3 and r["n_distinct_uuids"] == 3
    assert r["uuids_used_by_more_than_one_file"] == 0 and r["files_without_uuid_by_class"] == {"Cls_B": 2}
    assert r["copy_named_files"] == 1 and r["groups_with_a_copy_file"] == 1 and r["of_which_with_more_than_one_uuid"] == 1
    assert r["files_with_augmentation_suffix"] == 0
