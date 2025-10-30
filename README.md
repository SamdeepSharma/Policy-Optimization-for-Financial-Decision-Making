## Policy Optimization for Financial Decision-Making

This repository contains four analysis notebooks that implement a pipeline for
exploring LendingClub loan data, training a supervised predictive model (MLP),
training an offline RL agent (Discrete CQL) for a one-step decision problem,
and comparing their economic value.

The notebooks are:

- `task_1_preprocessing.ipynb` — EDA & preprocessing; saves processed datasets
  and fitted transformers to `data/`.
- `task_2_mlp.ipynb` — Builds, trains and evaluates an MLP using TensorFlow/Keras;
  saves the trained model and test predictions to `models/`.
- `task_3_cql.ipynb` — Builds an RL dataset (one-step MDP), trains a discrete
  CQL agent using `d3rlpy`, and saves the agent and reward scaler to `models/`.
- `task_4_analysis.ipynb` — Loads models and processed data, evaluates the
  MLP (AUC/F1) and compares estimated policy value for the learned RL agent
  versus baselines.

## High-level pipeline (what to run and in which order)

1. Prepare the environment and install dependencies (see below).
2. Place the raw LendingClub CSV (`accepted_2007_to_2018Q4.csv`) in the repo root
   (or update the `DATA_FILE` path in `task_1_preprocessing.ipynb`).
3. Run `task_1_preprocessing.ipynb` to generate processed inputs in `data/`.
4. Run `task_2_mlp.ipynb` to train the MLP and save `models/mlp_model.keras`.
5. Run `task_3_cql.ipynb` to train the CQL agent and save `models/cql_agent.d3`.
6. Run `task_4_analysis.ipynb` to reproduce the evaluation and comparison report.

## Environment & Requirements

Recommended Python version: 3.8–3.11 (tested with 3.8/3.9). For reproducibility,
use a virtual environment or Conda environment. A `requirements.txt` is provided
with the key runtime dependencies.

Important notes:
- `tensorflow` is used for the MLP (Task 2). On Windows, GPU installs of
  TensorFlow require matching CUDA/cuDNN versions — if you don't have CUDA,
  use the CPU wheel.
- `d3rlpy` is used for offline RL (Task 3). It relies on PyTorch; on Windows it
  is often easiest to install PyTorch via the official PyTorch channels (see
  troubleshooting below).

The provided `requirements.txt` contains the main Python packages used by the
notebooks. You may need to install or adjust platform-specific packages (e.g.
PyTorch) separately for best compatibility.

## Quick setup (PowerShell)

Open PowerShell (pwsh) in the repository root and run:

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer Conda (recommended for Windows because of PyTorch/CUDA):

```pwsh
conda create -n policyopt python=3.9 -y
conda activate policyopt
pip install -r requirements.txt
# Install PyTorch separately using the instructions at https://pytorch.org/
```

## Data

This project expects the LendingClub accepted loans CSV named
`accepted_2007_to_2018Q4.csv` in the repository root. You can download the
dataset directly from Kaggle:

- LendingClub Loan Data — https://www.kaggle.com/datasets/wordsforthewise/lending-club

Note: Kaggle requires an account to download datasets and you must follow
their terms of use. Alternatively, use any CSV with the same column names
referenced in `task_1_preprocessing.ipynb`.

If your raw data file is large, set `SAMPLE_SIZE` in `task_1_preprocessing.ipynb`
to a smaller integer for quick experiments.

## How to run the notebooks (non-interactive / reproducible)

The notebooks include top-level `main()` functions and can be executed using
Jupyter's `nbconvert`. From PowerShell run:

```pwsh
# Run preprocessing (Task 1)
jupyter nbconvert --to notebook --execute task_1_preprocessing.ipynb --ExecutePreprocessor.timeout=1200 --output task_1_preprocessing.executed.ipynb

# Train the MLP (Task 2)
jupyter nbconvert --to notebook --execute task_2_mlp.ipynb --ExecutePreprocessor.timeout=1200 --output task_2_mlp.executed.ipynb

# Train the CQL agent (Task 3)
jupyter nbconvert --to notebook --execute task_3_cql.ipynb --ExecutePreprocessor.timeout=3600 --output task_3_cql.executed.ipynb

# Final analysis (Task 4)
jupyter nbconvert --to notebook --execute task_4_analysis.ipynb --ExecutePreprocessor.timeout=1200 --output task_4_analysis.executed.ipynb
```

Notes:
- Increase the `--ExecutePreprocessor.timeout` value if you have slow hardware
  or if training is expected to take longer (especially for CQL).
- You can also open and run the notebooks interactively in JupyterLab or
  VS Code's notebook UI.

## Expected artifacts (after successful run)

- `data/X_train_final.pkl`, `data/y_train.pkl`, `data/X_test_final.pkl`, `data/y_test.pkl` (Task 1)
- `models/mlp_model.keras`, `models/mlp_test_pred_probs.npy` (Task 2)
- `models/cql_agent.d3`, `models/reward_scaler.pkl` (Task 3)
- Generated `*.executed.ipynb` files (if using `nbconvert` for reproducible runs)

## Troubleshooting & Tips

- d3rlpy / PyTorch issues on Windows: If `pip install d3rlpy` fails due to
  PyTorch wheel availability, install PyTorch first using the official
  installer from https://pytorch.org/ (select your CUDA version or CPU-only),
  then re-run `pip install d3rlpy`.
- If TensorFlow fails to install on your machine or you prefer PyTorch-based
  workflows, you could replace the MLP with a PyTorch model. The notebooks were
  written to be modular: Task 2 saves predictions and a model file that Task 4
  consumes.
- If the `task_3_cql.ipynb` training is too slow, reduce `n_steps` in the
  notebook config (the default in the code is 50,000 steps) or train on a
  smaller sample (set `SAMPLE_SIZE` in Task 1) for development.

## Reproducibility notes

- Seeds: Notebooks set `RANDOM_SEED = 42` and call NumPy/TensorFlow seed
  setters where relevant. Exact bitwise reproducibility may still vary across
  platforms and library versions.
- Savepoints: The preprocessing notebook saves fitted transformers
  (`scaler.pkl`, `ohe.pkl`, `num_imputer.pkl`, `cat_imputer.pkl`) so you can
  reuse them without retraining preprocessing steps.

## Contact & Next steps

If you'd like, I can:

- create a `run_all.sh` / PowerShell script that runs the four notebooks in the
  correct order and captures logs,
- produce a `Dockerfile` that pins an environment and installs PyTorch/TensorFlow
  for fully reproducible runs,
- or convert the notebooks to standalone Python scripts (e.g., `task_1_preprocessing.py`) for CI-friendly execution.

---

Author: Samdeep Sharma (102217183)

