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

ALGOS = {
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG
}


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate(model, env, n_episodes=100, seed=0):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        habs = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            habs.append(info.get("true_habitability", np.nan))
            done = term or trunc
            steps += 1

        results.append({
            "episode": ep,
            "return": total_r,
            "steps": steps,
            "mean_habitability": float(np.nanmean(habs))
        })

    return results


def main(args):
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

    log_dir = os.path.join(ROOT_DIR, "logs", "eval")
    plot_dir = os.path.join(ROOT_DIR, "plots")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    results = evaluate(model, env, n_episodes=args.n_episodes, seed=seed)
    csv_path = os.path.join(log_dir, f"{algo}_eval_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "steps", "mean_habitability"])
        writer.writeheader()
        writer.writerows(results)

    returns = [r["return"] for r in results]
    habs = [r["mean_habitability"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.plot(returns, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{algo} evaluation returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{algo}_eval_returns.png"))
    plt.close()

    # Habitability plot
    plt.figure(figsize=(6, 4))
    plt.plot(habs, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Mean habitability")
    plt.title(f"{algo} evaluation habitability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{algo}_eval_habitability.png"))
    plt.close()

    print(f"Saved CSV → {csv_path}")
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