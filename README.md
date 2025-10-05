# 🫧 boilers
![CI](https://github.com/acb-code/boilers/actions/workflows/ci.yml/badge.svg)


**boilers** is a personal deep learning workshop — a collection of modular implementations, experiments, and blog-style notebooks exploring topics in modern neural networks, optimization, and representation learning.

It’s the “engine room” of the broader ship:
> **barnacles** – general notes & sketches

> **boilers** – deep learning systems and implementations

> **compass** – reinforcement learning & navigation

---

## 🚀 Features

- Modular PyTorch-based code under `src/boilers/`
- Reusable layers, models, and training utilities
- Blog-style, self-contained notebooks in `notebooks/`
- Configurable experiment structure with logs and outputs in `experiments/`
- Clean reproducible environment setup with Conda

---

## Notebooks
#### 01 – Linear & Logistic Regression
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/acb-code/boilers/blob/main/notebooks/2025-10-05-intro/01_linear_logistic_mlp.ipynb
)


## 🧠 Getting Started

### 1️⃣ Create and activate the Conda environment

```bash
# Create a new environment named boilers
conda create -n boilers python=3.10 -y
conda activate boilers
```

### 2️⃣ Install dependencies
You can install via pip using the provided pyproject.toml and requirements.txt:

```bash
# Ensure pip + build tools are up to date
python -m pip install --upgrade pip setuptools wheel

# (Optional) Install extra packages if you plan to run notebooks
pip install jupyterlab ipykernel

# Install in editable mode so notebooks can import modules directly
pip install -e .
```

### Alternatively, if you prefer Conda-only dependency resolution:
```bash
conda install pytorch torchvision numpy matplotlib tqdm -c pytorch
pip install ipykernel
pip install -e .
```

## 🧪 Example Workflow
```bash
# 1. Clone
git clone https://github.com/<your-username>/boilers.git
cd boilers

# 2. Create environment
conda create -n boilers python=3.10 -y
conda activate boilers

# 3. Install
# install developer dependencies
pip install -r requirements.txt

# Install dev dependencies + local package
pip install -e ".[dev]"

# install basics
pip install -e .

# 4. Open a nb and start
```

## Tests
Run the tests created and placed in the tests directory automatically with:
```bash
pytest -q
```

## 🧹 Code Formatting & Pre-Commit Hooks

This repository uses [pre-commit](https://pre-commit.com/) to keep code clean and consistent.

Install and activate hooks once:
```bash
pip install pre-commit
pre-commit install
```

### 🛠 Using the Makefile

Common tasks are wrapped in a Makefile so you don’t need to remember long commands.

```bash
# See available targets
make help

# Install dev dependencies + local package (editable)
make dev

# Run unit tests
make test

# Lint without changing files
make lint

# Auto-format code (Ruff fix, Black, isort)
make fmt

# Launch JupyterLab from the project root
make lab

# Clean caches and build artifacts
make clean

# Install pre-commit hooks and run them on all files once
make precommit
```

### 🧹 Pre-commit usage

First run will usually reformat files and exit non-zero (that’s expected).

```bash
# Run hooks on the full repo
make precommit
# If files were modified:
git add -A
make precommit     # should pass clean now
```

# References
“Concepts and some exercises inspired by Understanding Deep Learning (S.J.D. Prince).”.
