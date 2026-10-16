### First attempt: similarity-based split (kept, and reported as failed)

The first 'clean' split grouped images by embedding similarity. It did **not** remove leakage: by the file-name check 63.1% of the original validation images have a same-source image in train, and still 54.2% of that split's validation images do. Its results stay in the tables above.

### Second protocol: grouped splits (construction and checks)

Images are linked into groups (union of: same UUID, same class + base file name, and for the leaf split also the authors' leaf id). Whole groups go to train / validation / test (target 70/15/15, stratified by class, seed 0). Input: 87867 files. Split fingerprints are committed in `splits/split_hashes.json` and re-checked on Kaggle.

| check | photo-grouped | leaf+photo-grouped (primary) |
|---|---|---|
| images train / val / test | 61409 / 13230 / 13228 | 61012 / 13414 / 13441 |
| groups (largest) | 45040 (16) | 17399 (132) |
| groups over 100 images | 0 | 3 |
| per-class val share (min-max) | 0.150-0.153 | 0.150-0.161 |
| per-class test share (min-max) | 0.150-0.153 | 0.150-0.170 |
| groups / UUIDs / base names shared across parts | 0 / 0 / 0 | 0 / 0 / 0 |
| leaf ids shared across parts | 5626 | 0 |
| test images with a leaf id | 10145 | 10312 |
| val images (with a leaf id) whose leaf is also in train | 96.6% of 10171 | 0.0% of 10294 |
| test images (with a leaf id) whose leaf is also in train | 96.6% of 10145 | 0.0% of 10312 |

**Leaf-map coverage: 76.6% of the files** (67317 of 87867) have a leaf id from the authors' map; the other 23.4% are grouped by photo only, so same-leaf leakage can remain for them even in the leaf split. Per-class coverage ranges 0.0%-100.0%. **7 of 38 classes have no leaf id at all** (Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot, Corn_(maize)___Common_rust_, Corn_(maize)___Northern_Leaf_Blight, Corn_(maize)___healthy, Grape___healthy, Squash___Powdery_mildew, Tomato___Tomato_mosaic_virus), so the leaf-mapped subset, and therefore the headline number, covers only the other 31 classes. The map is downloaded at run time (sha256 `3b4b253683c6...`) and is not committed.
Files without a UUID: 2384 (Corn_(maize)___Common_rust_: 2384); files named with ' copy': 2921.

### Evidence for the grouping key (naming check)

Original PlantVillage repository (GitHub API, 50289 files in 36 classes; tree truncated by GitHub: True): 49097 files have a UUID prefix and 49097 UUIDs are distinct (0 UUIDs are used by more than one file); 1192 files have no UUID (Corn_(maize)___Common_rust_: 1192); 0 files carry an augmentation suffix; 1366 files are named ' copy', and in 607 of 607 (class, base name) groups that contain a copy the files have more than one UUID.

Kaggle copy: 2384 files have no UUID (Corn_(maize)___Common_rust_: 2384); 0 UUIDs appear under more than one label and 0 under more than one base name. On the dataset's original split, the share of validation images with a group member in train is 63.1% (uuid_only (files without uuid kept alone)), 65.9% (class_and_base_name), 65.9% (union_of_both). The union key gives 45040 groups, the largest with 16 images (0 groups over 40).

### Results by protocol (one seed per grouped run)

Final-epoch metrics. Validation accuracy for 'clean' is the mean of 3 seeds; every other row is one run. **Headline number: the leaf-mapped test accuracy on the leaf-grouped split** (files covered by the authors' leaf map; same-leaf leakage is 0.0% there).

| backbone | protocol | val acc | test acc (all) | test acc, leaf-mapped files (n) | test acc, files without leaf id (n) |
|---|---|---|---|---|---|
| mobilenet_v2 | original split (dataset's own) | 0.9917 | - | - | - |
| mobilenet_v2 | similarity 'clean' split (failed attempt, 3 seeds) | 0.9909 | - | - | - |
| mobilenet_v2 | photo-grouped | not run yet | not run yet | not run yet | not run yet |
| mobilenet_v2 | leaf+photo-grouped (primary) | not run yet | not run yet | not run yet | not run yet |
| efficientnet_b3 | original split (dataset's own) | 0.9970 | - | - | - |
| efficientnet_b3 | similarity 'clean' split (failed attempt, 3 seeds) | 0.9976 | - | - | - |
| efficientnet_b3 | photo-grouped | not run yet | not run yet | not run yet | not run yet |
| efficientnet_b3 | leaf+photo-grouped (primary) | not run yet | not run yet | not run yet | not run yet |

The grouped runs have not been executed yet, so there is no headline number.

### Observations (generated from the numbers above)

- No grouped runs yet.
