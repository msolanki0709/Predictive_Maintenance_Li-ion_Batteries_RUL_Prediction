import pandas as pd
import numpy as np
import os

def load_battery_data(battery_id="B0005", base_path="data/raw/"):
    """Load NASA battery dataset and return train/test sets"""
    path = os.path.join(base_path, f"{battery_id}.csv")
    df = pd.read_csv(path)

    # Example: Use cycle vs capacity
    X = df[["Cycle_Index"]].values
    y = df["Capacity"].values

    split = int(0.5 * len(X))  # first 50% for training
    return X[:split], y[:split], X[split:], y[split:]
