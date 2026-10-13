# Decisions

Every choice, why, and the alternatives. Newest steps at the bottom.

## Step 0: environment and repo setup

| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| 0.1 | New standalone repo `agriscan-reimpl`, separate from any other project | Keeps history, secrets and files clean | Subfolder of an existing repo |
| 0.2 | Repo will be public on GitHub, **not pushed until the owner says so** | Kaggle can `git clone` a public repo; no zip upload | Private repo + token (risks secret leaks), zip upload (manual, error-prone) |
| 0.3 | `LEARNING_NOTES.md` and `INTERVIEW_QA.md` are git-ignored | Personal study material; must not go public by accident | Commit them (rejected by owner) |
| 0.4 | No local GPU (Intel Iris Xe only): real training on Kaggle GPU; laptop for code and CPU smoke tests | A full run on an i7-1355U CPU would take days | Colab (needs a `kaggle.json` upload for the dataset), paid cloud |
| 0.5 | PyTorch only | Owner's choice; one framework is easier to explain | PyTorch + TensorFlow as in the original (README states the original used both) |
| 0.6 | `requirements.txt` uses lower bounds, not exact pins; exact versions are saved in each results JSON | Kaggle/Colab ship their own torch; forcing a pin would reinstall ~GBs or break CUDA | `pip freeze` lockfile (breaks across platforms) |
| 0.7 | Every run writes JSON with seed, config, date, library versions, device, git commit via `save_results()` | Honesty rule: every reported metric must be traceable to a run | Printing numbers in notebooks (not reproducible, easy to misquote) |
| 0.8 | Smoke test uses random data and `weights=None` | Runs offline on CPU in seconds; makes no accuracy claim | Tiny real-data subset (added in step 1 once data is downloaded) |
| 0.9 | Local Python is 3.14.7; CPU torch 2.14.1 installed and verified to run | It is what is installed; `requires-python >= 3.10` keeps Kaggle (3.10+) compatible | Install an older Python (unnecessary, the smoke test passes) |
| 0.10 | Commits carry `Co-Authored-By: Claude`; README says Claude Code was used as a tutor | Owner wants to be open about it | Omit (rejected by owner) |

## Known limits of step 0

- `set_seed` makes runs comparable, not bit-identical on GPU (some CUDA kernels are non-deterministic).
- The Kaggle notebook has a `<YOUR_GITHUB_USER>` placeholder until the repo is pushed; it has **not** been run on Kaggle yet.
