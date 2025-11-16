import os
import random
import numpy as np
import torch
import csv
from stable_baselines3.common.monitor import Monitor


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_monitored_env(env_fn, log_dir, rank=0):
    os.makedirs(log_dir, exist_ok=True)
    env = env_fn()
    env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
    return env


class CSVWriter:
    """Simple CSV writer for episodic metrics."""

    def __init__(self, path, headers):
        self.path = path
        self.headers = headers
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def write(self, row):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
