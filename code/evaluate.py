import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, TD3, DDPG, PPO
from envs import make_env_fn
from utils import set_global_seeds
import yaml
import csv
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(BASE_DIR, "config.yaml")
ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}#, "PPO": PPO}


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def evaluate(model, env, n_episodes=10, seed=0):
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        done = False
        total_r = 0.0
        steps = 0
        habs = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # greedy/deterministic
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            habs.append(info.get("true_habitability", np.nan))
            done = term or trunc
            steps += 1
        results.append({"episode": ep, "return": total_r, "steps": steps, "mean_habitability": float(np.nanmean(habs))})
    return results

def main(args):
    config = load_config(args.config)
    set_global_seeds(config.get("seed", 0))
    max_steps = config["env"]["max_steps"]
    seed = config.get("seed", 0)

    algo = args.algo.upper()
    model_path = args.model
    if algo not in ALGOS:
        raise ValueError(algo)

    ModelClass = ALGOS[algo]
    model = ModelClass.load(model_path)

    env = make_env_fn(max_steps=max_steps, stable_steps_required=config["env"]["stable_steps_required"], seed=seed)()

    results = evaluate(model, env, n_episodes=args.n_episodes, seed=seed)
    os.makedirs("logs/eval", exist_ok=True)
    csv_path = f"logs/eval/{algo}_eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "return", "steps", "mean_habitability"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    returns = [r["return"] for r in results]
    habs = [r["mean_habitability"] for r in results]

    plt.figure(figsize=(6,4))
    plt.plot(returns, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{algo} evaluation returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{algo}_eval_returns.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(habs, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Mean habitability")
    plt.title(f"{algo} evaluation habitability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{algo}_eval_habitability.png")
    plt.close()

    print(f"Saved evaluation CSV to {csv_path} and plots to plots/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/config.yaml")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm name: SAC/TD3/DDPG/PPO")
    parser.add_argument("--model", type=str, required=True, help="Path to saved SB3 model (zip)")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
