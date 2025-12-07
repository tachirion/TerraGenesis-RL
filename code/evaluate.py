"""
TerraGenesis RL Evaluation Script

Evaluates trained RL models on the TerraGenesis environment.

Supported Algorithms:
 - SAC (Soft Actor-Critic)
 - TD3 (Twin Delayed DDPG)
 - DDPG (Deep Deterministic Policy Gradient)

Features:
 - Episode- and step-level logging to CSV
 - Plots for returns and mean habitability per episode
 - Reproducible evaluation via configurable seed
 - Works with any Stable-Baselines3 compatible model

Typical Workflow:
 1. Load config.yaml
 2. Load trained SB3 model (.zip)
 3. Build evaluation environment
 4. Evaluate for N episodes
 5. Save CSV logs and plots
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3, DDPG
from envs import make_env_fn
from utils import set_global_seeds
import yaml
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIG_DEFAULT = os.path.join(BASE_DIR, "config.yaml")

ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}


def load_config(path: str) -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    path : str
        Path to YAML config file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, env, n_episodes: int = 100, seed: int = 0):
    """
    Evaluate a trained RL model on the given environment.

    Parameters
    ----------
    model : BaseAlgorithm
        Trained Stable-Baselines3 model.
    env : gym.Env
        Environment instance to evaluate on.
    n_episodes : int
        Number of episodes to run.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (episode_results, step_results) where:
        - episode_results: list of dicts with total reward, mean habitability, instability, and resource used per episode
        - step_results: list of dicts containing step-level metrics
    """
    episode_results = []
    step_results = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        habs = []
        event_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)

            habitability = info.get("true_habitability", np.nan)
            event = info.get("event", "none")
            instability = info.get("instability", np.nan)
            resource_used = info.get("resource_used", 1)

            total_r += r
            habs.append(habitability)
            if event != "none":
                event_count += 1
            steps += 1
            done = term or trunc

            step_results.append({
                "episode": ep,
                "step": steps,
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "reward": r,
                "true_habitability": habitability,
                "event": event,
                "instability": instability,
                "resource_used": resource_used
            })

        episode_results.append({
            "episode": ep,
            "total_reward": total_r,
            "mean_habitability": float(np.nanmean(habs)) if habs else np.nan,
            "instability": event_count,
            "resource_used": steps
        })

    return episode_results, step_results


def main(args):
    """
    Main evaluation routine.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments parsed by argparse.
    """
    config = load_config(args.config)
    set_global_seeds(config.get("seed", 0))
    max_steps = config["env"]["max_steps"]
    seed = config.get("seed", 0)

    algo = args.algo.upper()
    if algo not in ALGOS:
        raise ValueError(f"Unsupported algo: {algo}")

    ModelClass = ALGOS[algo]
    model = ModelClass.load(args.model)

    env = make_env_fn(
        max_steps=max_steps,
        stable_steps_required=config["env"]["stable_steps_required"],
        seed=seed
    )()

    # Create directories for logs and plots
    log_dir = os.path.join(ROOT_DIR, "logs", "eval")
    plot_dir = os.path.join(ROOT_DIR, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    episode_results, step_results = evaluate(model, env, n_episodes=args.n_episodes, seed=seed)

    csv_path = os.path.join(log_dir, f"{algo}_eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["episode", "total_reward", "mean_habitability", "instability", "resource_used"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_results)

    step_csv_path = os.path.join(log_dir, f"{algo}_eval_steps.csv")
    with open(step_csv_path, "w", newline="") as f:
        step_fieldnames = ["episode", "step", "action", "reward", "true_habitability", "event", "instability", "resource_used"]
        writer = csv.DictWriter(f, fieldnames=step_fieldnames)
        writer.writeheader()
        writer.writerows(step_results)

    returns = [r["total_reward"] for r in episode_results]
    habs = [r["mean_habitability"] for r in episode_results]

    plt.figure(figsize=(6, 4))
    plt.plot(returns, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{algo} Evaluation Returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{algo}_eval_returns.png"))
    plt.close()

    # Plot mean habitability
    plt.figure(figsize=(6, 4))
    plt.plot(habs, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Mean Habitability")
    plt.title(f"{algo} Evaluation Habitability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{algo}_eval_habitability.png"))
    plt.close()

    print(f"Saved episode-level CSV → {csv_path}")
    print(f"Saved step-level CSV → {step_csv_path}")
    print(f"Saved plots → {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=CONFIG_DEFAULT)
    parser.add_argument("--algo", type=str, required=True, help="SAC / TD3 / DDPG")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
