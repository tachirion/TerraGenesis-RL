import os
import random
import numpy as np
import torch
import csv
from stable_baselines3.common.callbacks import BaseCallback


# ================================================================
#  GLOBAL SEEDING
# ================================================================
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


# ================================================================
#  CSV WRITER
# ================================================================
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


# ================================================================
#  EPISODE-LEVEL CSV LOGGING CALLBACK
# ================================================================
class CSVLoggingCallback(BaseCallback):
    """
    Logs metrics only at the END of each episode.
    Works correctly with VecMonitor, which stores episode metrics
    inside info["episode"] only when done=True.
    """

    def __init__(self, writer, verbose=0):
        super().__init__(verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])

        for info in infos:
            if "episode" in info:
                ep_info = info.get("episode", {})

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
    Logs per-step values. Every step is stored and dumped to a CSV
    at the end of training.
    """

    def __init__(self, step_csv_path, verbose=0):
        super().__init__(verbose)
        self.step_csv_path = step_csv_path
        self.step_logs = []

    def _on_step(self) -> bool:
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
