import os
import random
import numpy as np
import torch
import csv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


def set_global_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    try:
        from stable_baselines3.common.utils import set_random_seed as sb3_set_seed
        sb3_set_seed(seed)
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


class CSVLoggingCallback(BaseCallback):
    def __init__(self, writer, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.episode_rewards = []
        self.episode_infos = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        reward = self.locals.get("rewards", [0.0])[0]
        if "episode" in info:
            ep_info = info["episode"]
            # Save metrics
            self.writer.write([
                ep_info["r"],  # total reward
                ep_info.get("true_habitability", np.nan),
                ep_info.get("instability", np.nan),
                ep_info.get("resource_used", np.nan)
            ])
        return True
