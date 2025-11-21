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


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, env, n_episodes=100, seed=0):
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
    plt.title(f"{algo} evaluation returns")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{algo}_eval_returns.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(habs, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Mean habitability")
    plt.title(f"{algo} evaluation habitability")
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

# run     : python code\evaluate.py --config code\config.yaml --algo TD3 --model models/td3/td3_final --n_episodes 200
# or e.g. : python code\evaluate.py --config code\config.yaml --algo SAC --model models/sac/sac_final --n_episodes 200
# or e.g. : python code\evaluate.py --config code\config.yaml --algo DDPG --model models/ddpg/ddpg_final --n_episodes 200