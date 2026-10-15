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
