"""
TerraGenesis RL Plotting Utilities

This script provides functions to read CSV logs and generate learning curve plots
for different RL algorithms (SAC, TD3, DDPG, PPO) used in TerraGenesis experiments.

Features:
 - Read multiple monitor CSV files matching a pattern
 - Compute rolling mean of episode returns
 - Plot and save learning curves for multiple algorithms
 - Designed to work with SB3 VecMonitor / CSV logging outputs

Typical Usage:
 1. Prepare CSV logs from training (VecMonitor or custom logging)
 2. Call read_monitor_csvs() to load data
 3. Call plot_returns() with a dict of {algorithm_name: [file_paths]}
 4. Plots are saved to the specified path (default: plots/learning_returns.png)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def read_monitor_csvs(pattern):
    """
    Read all CSV files matching a glob pattern into pandas DataFrames.

    Parameters
    ----------
    pattern : str
        Glob pattern to match CSV files.

    Returns
    -------
    list of tuples
        Each tuple is (file_path, pandas.DataFrame) for successfully read files.
    """
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
    """
    Generate a learning curve plot showing episode returns for multiple algorithms.

    Parameters
    ----------
    all_algo_monitor_files : dict
        Dictionary of {algorithm_name: list_of_csv_files}.
    out_path : str
        Path to save the output plot (default: "plots/learning_returns.png").

    Notes
    -----
    - Computes rolling mean over 10 episodes.
    - Only includes CSVs containing the 'r' column.
    """
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    algo_files = {}
    for algo in ["SAC", "TD3", "DDPG", "PPO"]:
        files = glob.glob(f"logs/**/monitor_*.csv", recursive=True)
        algo_files[algo] = [f for f in files if algo.lower() in f.lower() or True]

    plot_returns(algo_files)
    print("Saved comparison plot to plots/learning_returns.png")
