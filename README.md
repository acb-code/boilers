# 🫧 boilers

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

## 🧩 Repository Structure

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

# References
“Concepts and some exercises inspired by Understanding Deep Learning (S.J.D. Prince).”.