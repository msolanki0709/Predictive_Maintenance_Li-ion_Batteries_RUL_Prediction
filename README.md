# ABC-SVR RUL Prediction of Lithium-Ion Batteries

This repository implements **Remaining Useful Life (RUL) prediction** for Lithium-ion batteries using **Support Vector Regression (SVR)** optimized by the **Artificial Bee Colony (ABC) algorithm**, based on NASA PCoE datasets.

## Structure
- `src/` → Core modules (data loading, features, SVR, ABC optimizer, evaluation)
- `experiments/` → Scripts for baseline SVR, ABC-SVR, and comparison
- `notebooks/` → Original Jupyter Notebook
- `results/` → Logs and plots

## Usage
```bash
pip install -r requirements.txt
python experiments/run_abc_svr.py
