"""
TerraGenesis RL Utilities

Helper functions and callbacks for training TerraGenesis RL experiments.

Key Features:
 - Global seeding for Python, NumPy, PyTorch, and Stable-Baselines3
 - CSVWriter for episodic metrics
 - CSVLoggingCallback for episode-level logging
 - StepLoggingCallback for per-step logging to CSV
 - Designed to integrate with VecMonitor and VecNormalize

Intended Usage:
 1. Call set_global_seeds(seed) before training
 2. Use CSVWriter to create a CSV file for logging
 3. Attach CSVLoggingCallback and/or StepLoggingCallback to SB3 model.learn()
 4. Logs can be used for analysis or visualization
"""

import os
import random
import numpy as np
import torch
import csv
from stable_baselines3.common.callbacks import BaseCallback


# ================================================================
#  GLOBAL SEEDING
# ================================================================
def set_global_seeds(seed: int):
    """
    Set global seeds for reproducibility across Python, NumPy, PyTorch, and Stable-Baselines3.

    Parameters
    ----------
    seed : int
        The seed value to use for all random generators.

    Notes
    -----
    - Ensures deterministic behavior where possible.
    - If torch is unavailable, silently continues.
    - Compatible with GPU or CPU PyTorch usage.
    """
    # Python built-in randomness
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # Stable-Baselines3 seeding
    try:
        from stable_baselines3.common.utils import set_random_seed as sb3_set_seed
        sb3_set_seed(seed)
    except Exception:
        pass


# ================================================================
#  CSV WRITER
# ================================================================
class CSVWriter:
    """
    Simple CSV writer for episodic metrics.

    Parameters
    ----------
    path : str
        Path to CSV file to write.
    headers : list of str
        List of column names for CSV header.

    Notes
    -----
    - Creates parent directories if they do not exist.
    - Writes headers automatically if file does not exist.
    """

    def __init__(self, path: str, headers: list):
        self.path = path
        self.headers = headers
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def write(self, row: list):
        """
        Append a single row of data to the CSV file.

        Parameters
        ----------
        row : list
            Row of values to write. Must match length/order of headers.
        """
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# ================================================================
#  EPISODE-LEVEL CSV LOGGING CALLBACK
# ================================================================
class CSVLoggingCallback(BaseCallback):
    """
    Logs episode-level metrics to a CSV file.

    Works with VecMonitor or custom environments that store
    metrics in info["episode"] at the end of each episode.
    """

    def __init__(self, writer: CSVWriter, verbose: int = 0):
        """
        Parameters
        ----------
        writer : CSVWriter
            CSVWriter instance to handle writing rows.
        verbose : int
            Verbosity level (0 = silent, 1 = print each episode record)
        """
        super().__init__(verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        """
        Called at every environment step.

        Extracts episode metrics from info dicts and writes them to CSV.

        Returns
        -------
        bool
            True to continue training.
        """
        infos = self.locals.get("infos", [{}])

        for info in infos:
            if "episode" in info:
                ep_info = info.get("episode", {})

                # Extract metrics
                r = ep_info.get("r", np.nan)
                hab = info.get("true_habitability", ep_info.get("true_habitability", np.nan))
                inst = info.get("instability", ep_info.get("instability", np.nan))
                res = info.get("resource_used", ep_info.get("resource_used", np.nan))

                self.writer.write([r, hab, inst, res])

                if self.verbose > 0:
                    print("[CSVLoggingCallback] Episode record:",
                          {"r": r, "true_habitability": hab, "instability": inst, "resource_used": res})

        return True


# ================================================================
#  PER-STEP CSV LOGGING CALLBACK
# ================================================================
class StepLoggingCallback(BaseCallback):
    """
    Logs per-step metrics to memory and writes them to CSV at
    the end of training.

    Useful for analyzing the progression of metrics within episodes.
    """

    def __init__(self, step_csv_path: str, verbose: int = 0):
        """
        Parameters
        ----------
        step_csv_path : str
            Path to save per-step CSV.
        verbose : int
            Verbosity level (0 = silent, 1 = print number of steps written)
        """
        super().__init__(verbose)
        self.step_csv_path = step_csv_path
        self.step_logs = []

    def _on_step(self) -> bool:
        """
        Called at every environment step.

        Collects step-level info from locals and stores in memory.
        """
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", [None] * len(infos))
        rewards = self.locals.get("rewards", [0.0] * len(infos))

        for i, info in enumerate(infos):
            action = actions[i] if i < len(actions) else None
            reward = rewards[i] if i < len(rewards) else 0.0

            ep_data = info.get("episode", {})
            step_entry = {
                "episode_reward": ep_data.get("r", np.nan),
                "step": info.get("step", np.nan),
                "action": (
                    action.tolist() if hasattr(action, "tolist") else action
                ),
                "reward": reward,
                "true_habitability": info.get("true_habitability", ep_data.get("true_habitability", np.nan)),
                "event": info.get("event", "none"),
                "instability": info.get("instability", ep_data.get("instability", np.nan)),
                "resource_used": info.get("resource_used", ep_data.get("resource_used", np.nan)),
            }

            self.step_logs.append(step_entry)

        return True

    def _on_training_end(self):
        """
        Called at the end of training.

        Writes all stored step logs to the CSV file.
        """
        os.makedirs(os.path.dirname(self.step_csv_path), exist_ok=True)

        fieldnames = [
            "episode_reward",
            "step",
            "action",
            "reward",
            "true_habitability",
            "event",
            "instability",
            "resource_used",
        ]

        with open(self.step_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.step_logs)

        if self.verbose > 0:
            print(f"[StepLoggingCallback] Wrote {len(self.step_logs)} step records to CSV")
