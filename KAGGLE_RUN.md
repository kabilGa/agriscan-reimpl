# Running the experiments on Kaggle (Phase B)

One notebook, "Run all". No Kaggle API token or password is needed anywhere: everything is done in the browser.

1. **Build the code bundle on your laptop** (commit first, so the results name an exact commit):
   `.venv\Scripts\python scripts/make_kaggle_bundle.py` creates `artifacts/agriscan_code.zip` (code only: no notes, weights, data or git history) and prints the commit it contains.
2. **Upload it as a private dataset:** kaggle.com -> Datasets -> New Dataset -> drop the zip, title `agriscan-code`, visibility **Private** -> Create.
3. **Open the notebook:** Code -> New Notebook -> File -> Import Notebook -> upload `notebooks/kaggle_run_all.ipynb`.
4. **Add two inputs** (right sidebar -> Add Input): your dataset `agriscan-code`, and the public dataset **New Plant Diseases Dataset**. Its licence terms are recorded in `LICENSES.md` ("Data files © Original Authors": no open licence, so the data stays on Kaggle and is never copied into the repo). The code finds the dataset folder itself and prints the path in the log; if it cannot, it stops with a message saying so.
5. **Settings:** Accelerator = **GPU T4 x2** (or P100). Internet = **On** (needs your phone-verified account; the code downloads ImageNet weights and the YOLO weights at run time).
6. **Run:** Save Version -> **Save & Run All (Commit)**. It keeps running if you close the browser. Each epoch prints its time; each stage prints `=== name: done`. The log's first cells also run the tests; if they fail, stop and tell me.
7. **Expected duration (an estimate, see the handoff message):** several hours, with a hard budget of 10 h: stages that would not fit are skipped and recorded in `results/run_status.json`. Core runs come first; ablations, seed replicates, YOLO, and extras follow.
8. **When it finishes** (Version -> Output tab) download: `export/agriscan_results.zip` (all JSON, the clean split, the pair galleries) and `export/models/mobilenet_v2__clean__s0__base.pt` (weights for the web app: keep it **outside git**).
9. **Put the results into the repo:** unzip `agriscan_results.zip` at the repo root, so `results/` and `splits/` appear. Do **not** copy `.pt` files or `kaggle.json` into the repo. Then tell me "results are in" and I do steps 7-8 (real numbers into the README).
10. **If the session died** before finishing: open the notebook again and re-run all (finished stages are skipped, and an interrupted epoch resumes from its checkpoint if the same session is alive). If the whole session is gone, add the previous version's output as an input and set `RESUME_FROM` in the last-but-one cell.

Check also: in the log, `dataset root:` should name the plant dataset (not your code dataset).

## Notes from the first real run (2026-10-15)

- **Code under test:** commit `ace8631` (recorded as `git_commit` in every results JSON). Wall time about 5.8 h on a Tesla T4, 20 of 20 stages done.
- **Cell 2 (`find_code`):** the run used a version that searches up to **3 folder levels** under `/kaggle/input` (for `pyproject.toml` and `agriscan*.zip`), because Kaggle mounted the inputs deeper than the original two-level glob, under `/kaggle/input/datasets/<owner>/<name>/`. The notebook in this repo now contains the same change (re-written from the description of that edit, not copied from the Kaggle notebook). Per the owner, the code under test was still commit `ace8631`; I could not inspect the Kaggle notebook itself, but all results JSON carry `git_commit` `ace8631` (and none carry `+dirty`).
- **Dataset path:** the dataset root resolved to `/kaggle/input/datasets`, so the paths in `splits/clean_split.csv.gz` are relative to that folder and start with `vipoooool/new-plant-diseases-dataset/...`. To reuse the split elsewhere, pass a `--root` whose subfolder `vipoooool/new-plant-diseases-dataset/` holds the dataset.
- **Delivery hiccups, for next time:** unzip `agriscan_results.zip` *into* the repo root (not into a new subfolder), and rename the downloaded checkpoint from `.zip` to `.pt` (it is a normal PyTorch file; Kaggle's download renamed it).
