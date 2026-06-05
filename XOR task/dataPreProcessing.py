import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

from functions import set_seed

# def make_XOR_data(n_samples, noise, seed):
#     set_seed(seed)
#     # Generate XOR data
#     X = np.random.rand(n_samples, 2) * 2 - 1  # Uniformly distributed in [-1, 1]
#     t = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)  # XOR labels

#     # Add noise
#     X += np.random.normal(0, noise, X.shape)

#     return X, t

# Make XOR data with skewed corners function
def make_XOR_data(N, noise, seed, corner_ratios=None):
    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Default skew: 80% -> 11, 10% -> 10, 9% -> 01, 1% -> 00
    if corner_ratios is None:
        corner_ratios = {"11": 0.80, "10": 0.10, "01": 0.09, "00": 0.01}
    if corner_ratios == "Equal":
        corner_ratios = {"11": 0.25, "10": 0.25, "01": 0.25, "00": 0.25}

    # Make exact counts (last class gets the remainder)
    n11 = int(corner_ratios["11"] * N)  # 0.80 * N
    n10 = int(corner_ratios["10"] * N)  # 0.10 * N
    n01 = int(corner_ratios["01"] * N)  # 0.09 * N
    n00 = N - n11 - n10 - n01           # whatever is left (should be close to 0.01 * N)

    # 
    corners = np.concatenate(
        [
            np.tile(np.array([1, 1], dtype=np.int64), (n11, 1)),
            np.tile(np.array([1, 0], dtype=np.int64), (n10, 1)),
            np.tile(np.array([0, 1], dtype=np.int64), (n01, 1)),
            np.tile(np.array([0, 0], dtype=np.int64), (n00, 1)),
        ],
        axis=0,
    )
    rng.shuffle(corners)

    # Sample uniformly inside each selected quadrant
    low = np.where(corners == 0, 0.0, 0.5)
    high = np.where(corners == 0, 0.5, 1.0)
    X = rng.uniform(low, high).astype(np.float32)

    # Optional jitter, clamped back to the original quadrant
    if noise > 0:
        X = X + rng.normal(0, noise, size=X.shape).astype(np.float32)
        # eps = 1e-6
        # x0_is_one = corners[:, 0] == 1
        # x1_is_one = corners[:, 1] == 1
        # X[~x0_is_one, 0] = np.clip(X[~x0_is_one, 0], 0.0, 0.5 - eps)
        # X[x0_is_one, 0] = np.clip(X[x0_is_one, 0], 0.5 + eps, 1.0)
        # X[~x1_is_one, 1] = np.clip(X[~x1_is_one, 1], 0.0, 0.5 - eps)
        # X[x1_is_one, 1] = np.clip(X[x1_is_one, 1], 0.5 + eps, 1.0)

    # XOR labels from the corner bits
    y = (corners[:, 0] ^ corners[:, 1]).astype(np.float32).reshape(-1, 1)

    return X, y, corners

def trainValTest_split(X, t, train_size=0.7418693231760913, val_size=0.2471432757105186, test_size=0.010987401113389979, seed=0):
    # First split into train and temp (val+test)
    X_trainVal, X_test, t_trainVal, t_test = train_test_split(X, t, test_size=test_size, random_state=seed)

    # Then split the temp into val and test
    val_relative_size = val_size / (val_size + train_size)  # Relative size

    X_train, X_val, t_train, t_val = train_test_split(X_trainVal, t_trainVal, test_size=val_relative_size, random_state=seed)

    return X_train, t_train, X_val, t_val, X_test, t_test