import os
import argparse
import yaml
import time
from stable_baselines3 import SAC, TD3, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import numpy as np

from envs import make_env_fn
from utils import set_global_seeds, make_monitored_env, CSVWriter


ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG} #, "PPO": PPO}

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_vec_envs(config, algo_name):
    n_envs = config["training"].get("n_envs", 1)
    seed = config.get("seed", 0)
    env_fns = []
    for i in range(n_envs):
        env_fns.append(make_env_fn(
            max_steps=config["env"]["max_steps"],
            stable_steps_required=config["env"]["stable_steps_required"],
            seed=seed + i
        ))
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env

def build_model(algo_name, vec_env, config, tb_log_name):
    algo = algo_name.upper()
    if algo not in ALGOS:
        raise ValueError(algo_name)
    ModelClass = ALGOS[algo]

    common_kwargs = {"verbose": 1}

    if algo == "SAC":
        kwargs = {
            "learning_rate": float(config["sac"]["learning_rate"]),
            "buffer_size": int(config["sac"]["buffer_size"]),
            "batch_size": int(config["sac"]["batch_size"]),
            "gamma": float(config["sac"]["gamma"]),
            "tau": float(config["sac"]["tau"]),
            "learning_starts": int(config["sac"]["learning_starts"]),
            "policy_kwargs": dict(net_arch=config["sac"]["net_arch"]),
            "ent_coef": config["sac"]["ent_coef"],
            **common_kwargs
        }
    elif algo == "TD3":
        kwargs = {
            "learning_rate": float(config["td3"]["learning_rate"]),
            "buffer_size": int(config["td3"]["buffer_size"]),
            "batch_size": int(config["td3"]["batch_size"]),
            "gamma": float(config["td3"]["gamma"]),
            "tau": float(config["td3"]["tau"]),
            "policy_delay": int(config["td3"]["policy_delay"]),
            "target_policy_noise": float(config["td3"]["target_policy_noise"]),
            "target_noise_clip": float(config["td3"]["target_noise_clip"]),
            "learning_starts": int(config["td3"]["learning_starts"]),
            "policy_kwargs": dict(net_arch=config["td3"]["net_arch"]),
            **common_kwargs
        }
    elif algo == "DDPG":
        kwargs = {
            "learning_rate": float(config["ddpg"]["learning_rate"]),
            "buffer_size": int(config["ddpg"]["buffer_size"]),
            "batch_size": int(config["ddpg"]["batch_size"]),
            "gamma": float(config["ddpg"]["gamma"]),
            "tau": float(config["ddpg"]["tau"]),
            "learning_starts": int(config["ddpg"]["learning_starts"]),
            "policy_kwargs": dict(net_arch=config["ddpg"]["net_arch"]),
            **common_kwargs
        }
    # elif algo == "PPO":
    #     kwargs = {
    #         "learning_rate": float(config["ppo"]["learning_rate"]),
    #         "n_steps": int(config["ppo"]["n_steps"]),
    #         "batch_size": int(config["ppo"]["batch_size"]),
    #         "n_epochs": int(config["ppo"]["n_epochs"]),
    #         "gamma": float(config["ppo"]["gamma"]),
    #         "gae_lambda": float(config["ppo"]["gae_lambda"]),
    #         "clip_range": float(config["ppo"]["clip_range"]),
    #         "policy_kwargs": dict(net_arch=config["ppo"]["net_arch"]),
    #         **common_kwargs
    #     }
    else:
        kwargs = common_kwargs

    model = ModelClass("MlpPolicy", vec_env, tensorboard_log=tb_log_name, **kwargs)
    return model

def main(args):
    config = load_config(args.config)
    set_global_seeds(config.get("seed", 0))

    results_dir = os.path.abspath(config["logging"]["results_dir"])
    tb_folder = os.path.join(results_dir, config["logging"]["tb_log_folder"])
    csv_folder = os.path.join(results_dir, config["logging"]["csv_log_folder"])
    os.makedirs(tb_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    algos = config["training"]["algorithms"]
    timesteps = int(config["training"]["timesteps"])
    save_freq = int(config["training"]["save_freq"])

    for algo_name in algos:
        print(f"\n=== Training {algo_name} ===")
        vec_env = build_vec_envs(config, algo_name)
        tb_log_name = os.path.join(tb_folder, algo_name)
        model = build_model(algo_name, vec_env, config, tb_log_name)

        checkpoint_dir = os.path.join("models", algo_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_cb = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_dir,
                                           name_prefix=f"{algo_name}_ckpt")

        # optional eval callback (left basic — you can add an EvalEnv if desired)
        # eval_env = build_vec_envs(config, algo_name)  # could reuse a separate env for eval
        # eval_cb = EvalCallback(eval_env, best_model_save_path=checkpoint_dir, eval_freq=save_freq,
        #                        n_eval_episodes=config["training"]["eval_episodes"], deterministic=True)

        start = time.time()
        model.learn(total_timesteps=timesteps, callback=[checkpoint_cb])
        elapsed = time.time() - start
        print(f"Finished {algo_name} training in {elapsed/60:.2f} minutes")

        final_path = os.path.join(checkpoint_dir, f"{algo_name}_final")
        model.save(final_path)
        vec_env.close()

        print(f"Saved model to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="code/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    main(args)
