#!/usr/bin/env python3
"""
train.py

Improved training script for TerraGenesis RL experiments.
Supports SAC, TD3, DDPG. Adds:
 - optional VecNormalize (obs / reward normalization)
 - learning rate schedule
 - OU / Normal action noise with linear annealing
 - EvalCallback + best-model saving
 - saving VecNormalize statistics
 - SAC: support explicit ent_coef (alpha) and target_entropy_scale
"""

import os
import argparse
import yaml
import time
import gymnasium as gym
import numpy as np
import torch.nn as nn

from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import get_linear_fn

from envs import make_env_fn
from utils import set_global_seeds, CSVWriter, CSVLoggingCallback, StepLoggingCallback

ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}


# --------------------------------------------------------------
#  LOAD CONFIG
# --------------------------------------------------------------
def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------
#  BUILD VECTORIZED ENVS
# --------------------------------------------------------------
def build_vec_envs(config: dict):
    """
    Build DummyVecEnv -> VecMonitor -> (optional) VecNormalize
    """
    n_envs = int(config.get("training", {}).get("n_envs", 1))
    seed = int(config.get("seed", 0))

    env_fns = [
        make_env_fn(
            max_steps=int(config["env"]["max_steps"]),
            stable_steps_required=int(config["env"].get("stable_steps_required", 0)),
            seed=seed + i,
        )
        for i in range(n_envs)
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Optional normalization (recommended for DDPG/TD3 and often helpful for SAC)
    if config.get("env", {}).get("normalize", False):
        norm_reward = bool(config.get("env", {}).get("norm_reward", False))
        clip_obs = float(config.get("env", {}).get("clip_obs", 10.0))
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_reward, clip_obs=clip_obs)

    return vec_env


# --------------------------------------------------------------
#  NOISE ANNEALING CALLBACK
# --------------------------------------------------------------
class NoiseAnnealingCallback(BaseCallback):
    """
    Linearly anneal action noise sigma from start_sigma to end_sigma
    over anneal_timesteps steps. Works for NormalActionNoise and OU noise.
    """
    def __init__(self, start_sigma: float, end_sigma: float, anneal_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)
        self.anneal_timesteps = max(1, int(anneal_timesteps))
        self.total_steps = 0

    def _on_step(self) -> bool:
        # increment and update noise sigma
        self.total_steps += 1
        self._update_noise()
        return True

    def _update_noise(self):
        if not hasattr(self.model, "action_noise") or self.model.action_noise is None:
            return
        frac = min(1.0, float(self.total_steps) / float(self.anneal_timesteps))
        new_sigma = float(self.start_sigma + frac * (self.end_sigma - self.start_sigma))

        # apply new sigma to noise object if possible
        try:
            noise = self.model.action_noise
            if hasattr(noise, "sigma"):
                noise.sigma[:] = new_sigma
            elif hasattr(noise, "std"):
                noise.std[:] = new_sigma
        except Exception:
            pass


# --------------------------------------------------------------
#  BUILD MODEL
# --------------------------------------------------------------
def build_model(algo_name: str, vec_env, config: dict, tb_log_name: str):
    """
    Build SB3 model with safe defaults and some algorithm-specific options.
    Specifically: for SAC we support ent_coef (alpha) and target_entropy_scale.
    """
    algo = algo_name.upper()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo: {algo}")

    ModelClass = ALGOS[algo]
    algo_cfg = config.get(algo.lower(), {}) or {}

    # Safe defaults & casting
    batch_size = int(algo_cfg.get("batch_size", 256))
    buffer_size = int(algo_cfg.get("buffer_size", 1_000_000))
    gamma = float(algo_cfg.get("gamma", 0.99))
    tau = float(algo_cfg.get("tau", 0.005))
    learning_starts = int(algo_cfg.get("learning_starts", 5000))
    lr_conf = algo_cfg.get("learning_rate", 1e-4)

    # Learning rate schedule (callable) if requested
    if bool(algo_cfg.get("lr_schedule", True)):
        try:
            lr = get_linear_fn(float(lr_conf), float(lr_conf) * 0.05, end_fraction=float(algo_cfg.get("lr_schedule_end_frac", 0.8)))
        except Exception:
            lr = float(lr_conf)
    else:
        lr = float(lr_conf)

    # Validate action space for action_dim (used by SAC target_entropy calc)
    action_dim = None
    try:
        action_space = vec_env.action_space
    except Exception:
        action_space = None

    if isinstance(action_space, gym.spaces.Box):
        action_dim = int(np.prod(action_space.shape))

    # Configure action noise only for DDPG/TD3 and when action space is continuous
    action_noise = None
    if algo in ("DDPG", "TD3") and action_dim is not None:
        noise_sigma = float(algo_cfg.get("noise_sigma", 0.2))  # starting sigma
        noise_type = str(algo_cfg.get("noise_type", "ou")).lower()
        if noise_type == "ou":
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim),
                                                        sigma=noise_sigma * np.ones(action_dim))
        else:
            action_noise = NormalActionNoise(mean=np.zeros(action_dim),
                                             sigma=noise_sigma * np.ones(action_dim))

    # Policy architecture
    net_arch = algo_cfg.get("net_arch", [400, 300])
    if isinstance(net_arch, str):
        try:
            net_arch = eval(net_arch)
        except Exception:
            net_arch = [400, 300]
    policy_kwargs = {"net_arch": net_arch, "activation_fn": nn.ReLU, "normalize_images": False}

    # Prepare algorithm-specific constructor args (SAC accepts ent_coef and target_entropy)
    extra_kwargs = {}
    if algo == "SAC":
        # ent_coef can be 'auto' or a float
        ent_conf = algo_cfg.get("ent_coef", "auto")
        if isinstance(ent_conf, str) and ent_conf.lower() == "auto":
            ent_coef = "auto"
        else:
            # try cast to float
            try:
                ent_coef = float(ent_conf)
            except Exception:
                ent_coef = "auto"
        extra_kwargs["ent_coef"] = ent_coef

        # target_entropy_scale: if present, compute target_entropy = -action_dim * scale
        scale = algo_cfg.get("target_entropy_scale", None)
        if scale is not None and action_dim is not None:
            try:
                scale = float(scale)
                target_entropy = -float(action_dim) * scale
                extra_kwargs["target_entropy"] = target_entropy
            except Exception:
                pass  # ignore if invalid

    # Build common model kwargs
    model_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        learning_starts=learning_starts,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log_name,
    )

    # attach action noise where applicable
    if action_noise is not None:
        model_kwargs["action_noise"] = action_noise

    # merge extra kwargs (SAC ent_coef / target_entropy)
    model_kwargs.update(extra_kwargs)

    model = ModelClass(**model_kwargs)
    return model


# --------------------------------------------------------------
#  MAIN TRAINING LOOP
# --------------------------------------------------------------
def main(args, algo_name):
    config = load_config(args.config)
    set_global_seeds(int(config.get("seed", 0)))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

    results_dir = os.path.join(ROOT_DIR, config.get("logging", {}).get("results_dir", "logs"))
    tb_folder = os.path.join(results_dir, config.get("logging", {}).get("tb_log_folder", "tensorboard"))
    csv_folder = os.path.join(results_dir, config.get("logging", {}).get("csv_log_folder", "csv"))
    os.makedirs(tb_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    algos = config.get("training", {}).get("algorithms", [])
    if algo_name:
        algos = [algo_name]

    timesteps = int(str(config.get("training", {}).get("timesteps", 1_000_000)).replace("_", ""))
    save_freq = int(str(config.get("training", {}).get("save_freq", 100_000)).replace("_", ""))

    for algo_name in algos:
        print(f"\n=== Training {algo_name} ===")

        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

        tb_log_name = os.path.join(tb_folder, f"{algo_name}_{run_id}")
        csv_path = os.path.join(csv_folder, f"{algo_name}_{run_id}.csv")
        step_csv_path = os.path.join(csv_folder, f"{algo_name}_{run_id}_steps.csv")

        # --------------------------------------------------
        # CSV episode logger
        # --------------------------------------------------
        csv_writer = CSVWriter(
            csv_path,
            headers=[
                "episode_reward",
                "true_habitability",
                "instability",
                "resource_used",
            ]
        )
        episode_cb = CSVLoggingCallback(csv_writer)
        step_cb = StepLoggingCallback(step_csv_path)

        # Build training vectorized envs
        vec_env = build_vec_envs(config)

        # Build model
        model = build_model(algo_name, vec_env, config, tb_log_name)

        # Prepare checkpoint directory
        checkpoint_dir = os.path.join(models_dir, algo_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{algo_name}_ckpt"
        )

        # -------------------------
        # Build evaluation env *same wrappers & normalization*
        # -------------------------
        eval_env = build_vec_envs(config)

        # If training env uses VecNormalize, copy running stats and set eval to non-training mode
        try:
            if isinstance(vec_env, VecNormalize) and isinstance(eval_env, VecNormalize):
                eval_env.training = False
                eval_env.norm_reward = vec_env.norm_reward
                eval_env.obs_rms = vec_env.obs_rms
                eval_env.ret_rms = vec_env.ret_rms
        except Exception:
            pass

        # Evaluation callback (works with vectorized eval_env)
        eval_freq = int(config.get("training", {}).get("eval_freq", 50_000))
        n_eval_episodes = int(config.get("training", {}).get("eval_episodes", 5))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=checkpoint_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )

        # Noise annealing (only used if model has action_noise)
        ddpg_cfg = config.get("ddpg", {}) or {}
        start_sigma = float(ddpg_cfg.get("noise_sigma", 0.2))
        end_sigma = float(ddpg_cfg.get("noise_sigma_end", 0.02))
        anneal_timesteps = int(str(config.get("training", {}).get("timesteps", timesteps)).replace("_", ""))
        noise_cb = NoiseAnnealingCallback(start_sigma, end_sigma, anneal_timesteps)

        # Configure logger to write stdout + tensorboard
        new_logger = configure(tb_log_name, ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        # Combined callbacks
        callback_list = [checkpoint_cb, episode_cb, step_cb, eval_callback, noise_cb]

        # ------------------------
        # TRAIN
        # ------------------------
        start = time.time()
        model.learn(
            total_timesteps=timesteps,
            callback=callback_list,
        )
        duration_min = (time.time() - start) / 60.0
        print(f"Training completed in {duration_min:.2f} minutes")

        # Save final model
        final_path = os.path.join(checkpoint_dir, f"{algo_name}_final")
        model.save(final_path)

        # If we used VecNormalize, save its stats so we can load later
        try:
            if isinstance(vec_env, VecNormalize):
                vnorm_path = os.path.join(checkpoint_dir, f"{algo_name}_vecnormalize.pkl")
                vec_env.save(vnorm_path)
                print(f"Saved VecNormalize to {vnorm_path}")
        except Exception:
            pass

        # Close envs
        try:
            vec_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass

        print(f"Saved model to {final_path}")
        print(f"TensorBoard logs: {tb_log_name}")
        print(f"Episode CSV log: {csv_path}")
        print(f"Step CSV log: {step_csv_path}")


# --------------------------------------------------------------
#  ENTRY POINT
# --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--algo", type=str, help="Optional: SAC, TD3, or DDPG")
    args = parser.parse_args()
    main(args, args.algo)
