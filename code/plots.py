import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def read_monitor_csvs(pattern):
    files = glob.glob(pattern)
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, comment='#')
            frames.append((f, df))
        except Exception as e:
            print("Skipping", f, ":", e)
    return frames

def plot_returns(all_algo_monitor_files, out_path="plots/learning_returns.png"):
    plt.figure(figsize=(8,5))
    for algo, files in all_algo_monitor_files.items():
        returns = []
        for f in files:
            df = pd.read_csv(f, comment='#')
            if "r" in df.columns:
                returns.extend(df["r"].values.tolist())
        if len(returns) > 0:
            plt.plot(pd.Series(returns).rolling(10, min_periods=1).mean(), label=algo)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return (rolling mean)")
    plt.legend()
    plt.title("Learning curves (returns)")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    algo_files = {}
    for algo in ["SAC","TD3","DDPG","PPO"]:
        pattern = f"logs/csv/**/{algo}_*.csv"
        files = glob.glob(f"logs/**/monitor_*.csv", recursive=True)
        algo_files[algo] = [f for f in files if algo.lower() in f.lower() or True]
    plot_returns(algo_files)
    print("Saved comparison plot to plots/learning_returns.png")
