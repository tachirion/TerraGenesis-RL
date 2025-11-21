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

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for info in infos:
            if "episode" in info:
                ep_info = info["episode"]
                self.writer.write([
                    ep_info.get("r", np.nan),
                    ep_info.get("true_habitability", np.nan),
                    ep_info.get("instability", np.nan),
                    ep_info.get("resource_used", np.nan)
                ])
        return True


class StepLoggingCallback(BaseCallback):
    def __init__(self, step_csv_path, verbose=0):
        super().__init__(verbose)
        self.step_csv_path = step_csv_path
        self.step_logs = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", [None] * len(infos))
        rewards = self.locals.get("rewards", [0] * len(infos))

        for i, info in enumerate(infos):
            action = actions[i] if i < len(actions) else None
            reward = rewards[i] if i < len(rewards) else 0
            self.step_logs.append({
                "episode": info.get("episode", np.nan),
                "step": info.get("step", np.nan),
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": reward,
                "true_habitability": info.get("true_habitability", np.nan),
                "event": info.get("event", "none"),
                "instability": info.get("instability", np.nan),
                "resource_used": info.get("resource_used", 1)
            })
        return True

    def _on_training_end(self):
        os.makedirs(os.path.dirname(self.step_csv_path), exist_ok=True)
        fieldnames = ["episode","step","action","reward","true_habitability","event","instability","resource_used"]
        import csv
        with open(self.step_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.step_logs)

