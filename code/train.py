import os
import argparse
import yaml
import time
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from envs import make_env_fn
from utils import set_global_seeds, CSVWriter, CSVLoggingCallback

ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_vec_envs(config):
    n_envs = config["training"].get("n_envs", 1)
    seed = config.get("seed", 0)

    env_fns = [
        make_env_fn(
            max_steps=config["env"]["max_steps"],
            stable_steps_required=config["env"]["stable_steps_required"],
            seed=seed + i,
        )
        for i in range(n_envs)
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env


def build_model(algo_name, vec_env, config, tb_log_name):
    algo = algo_name.upper()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo: {algo}")

    ModelClass = ALGOS[algo]
    common_kwargs = {"verbose": 1}

    algo_cfg = config[algo.lower()]

    kwargs = {
        "learning_rate": float(algo_cfg["learning_rate"]),
        "buffer_size": int(algo_cfg["buffer_size"]),
        "batch_size": int(algo_cfg["batch_size"]),
        "gamma": float(algo_cfg["gamma"]),
        "tau": float(algo_cfg["tau"]),
        "learning_starts": int(algo_cfg["learning_starts"]),
        "policy_kwargs": dict(net_arch=algo_cfg["net_arch"]),
        **common_kwargs,
    }

    model = ModelClass(
        "MlpPolicy",
        vec_env,
        tensorboard_log=tb_log_name,
        **kwargs
    )
    return model


def main(args, algo_name):
    config = load_config(args.config)
    set_global_seeds(config.get("seed", 0))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

    results_cfg = config.get("logging", {}).get("results_dir", "results")
    if os.path.isabs(results_cfg):
        results_dir = os.path.abspath(results_cfg)
    else:
        results_dir = os.path.abspath(os.path.join(ROOT_DIR, results_cfg))

    fs_root = os.path.abspath(os.sep)
    if os.path.normcase(os.path.normpath(results_dir)) == os.path.normcase(fs_root):
        print(f"Warning: resolved results_dir is filesystem root ({fs_root}); switching to project-local 'results' folder.")
        results_dir = os.path.join(ROOT_DIR, "results")

    tb_folder = os.path.join(results_dir, config["logging"]["tb_log_folder"])
    csv_folder = os.path.join(results_dir, config["logging"]["csv_log_folder"])
    os.makedirs(tb_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    algos = config["training"]["algorithms"]
    if algo_name:
        if algo_name not in algos:
            raise ValueError(f"{algo_name} not in config training algorithms: {algos}")
        algos = [algo_name]

    timesteps = int(config["training"]["timesteps"])
    save_freq = int(config["training"]["save_freq"])

    for algo_name in algos:
        print(f"\n=== Training {algo_name} ===")

        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
        tb_log_name = os.path.join(tb_folder, f"{algo_name}_{run_id}")
        csv_path = os.path.join(csv_folder, f"{algo_name}_{run_id}.csv")

        csv_writer = CSVWriter(
            csv_path,
            headers=["episode", "total_reward", "mean_habitability", "instability", "resource_used"]
        )

        vec_env = build_vec_envs(config)
        model = build_model(algo_name, vec_env, config, tb_log_name)

        checkpoint_dir = os.path.join(models_dir, algo_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{algo_name}_ckpt"
        )

        csv_cb = CSVLoggingCallback(csv_writer)

        print("TensorBoard folder:", tb_folder)
        print("TensorBoard run path:", tb_log_name)
        os.makedirs(tb_log_name, exist_ok=True)

        new_logger = configure(tb_log_name, ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        start = time.time()
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_cb, csv_cb],
        )
        print(f"Training completed in {(time.time() - start)/60:.2f} minutes")

        final_path = os.path.join(checkpoint_dir, f"{algo_name}_final")
        model.save(final_path)
        vec_env.close()

        print(f"Saved model to {final_path}")
        print(f"TensorBoard logs: {tb_log_name}")
        print(f"CSV log saved: {csv_path}")
        print(f"Run with:\n  tensorboard --logdir {tb_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--algo", type=str, help="Optional: specific algorithm to train (SAC, TD3, DDPG)")
    args = parser.parse_args()
    main(args, args.algo)


# run     : python train.py --config config.yaml
# or e.g. : python train.py --config config.yaml --algo SAC