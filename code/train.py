"""
Training script for TerraGenesis RL experiments using Stable-Baselines3.

Supported algorithms:
 - SAC (Soft Actor-Critic)
 - TD3 (Twin Delayed DDPG)
 - DDPG (Deep Deterministic Policy Gradient)

Features:
 - Optional VecNormalize for observation and reward normalization
 - Linear learning rate schedule
 - Action noise (OU / Normal) with linear annealing
 - Evaluation callback with best-model saving
 - CSV and TensorBoard logging
 - SAC support for explicit ent_coef (alpha) and target_entropy scaling
 - Safe handling of config parameters and environment seeding

Workflow:
 1. Load config.yaml
 2. Build vectorized training environment
 3. Build model (with algorithm-specific options)
 4. Setup evaluation environment (mirroring normalization)
 5. Configure callbacks: CSV logging, checkpointing, evaluation, noise annealing
 6. Train for total_timesteps
 7. Save final model and VecNormalize statistics (if used)
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import get_linear_fn
from envs import make_env_fn
from utils import set_global_seeds, CSVWriter, CSVLoggingCallback, StepLoggingCallback

ALGOS = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}


# --------------------------------------------------------------
#  LOAD CONFIGURATION
# --------------------------------------------------------------
def load_config(path: str) -> dict:
    """
    Load training configuration from YAML file.

    Parameters
    ----------
    path : str
        Path to YAML config file.

    Returns
    -------
    dict
        Configuration dictionary parsed from YAML.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------
#  BUILD VECTORIZED ENVIRONMENTS
# --------------------------------------------------------------
def build_vec_envs(config: dict):
    """
    Build a vectorized environment for training.

    Pipeline:
        DummyVecEnv -> VecMonitor -> (optional) VecNormalize

    Parameters
    ----------
    config : dict
        Configuration dictionary with environment parameters.

    Returns
    -------
    vec_env : VecEnv
        Vectorized and optionally normalized environment.
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
    Linearly anneals the action noise standard deviation from start_sigma
    to end_sigma over anneal_timesteps. Supports NormalActionNoise and
    Ornstein-Uhlenbeck noise used by TD3/DDPG.

    Usage: attach as a callback to SB3 model.learn()
    """
    def __init__(self, start_sigma: float, end_sigma: float, anneal_timesteps: int, verbose: int = 0):
        """
        Parameters
        ----------
        start_sigma : float
            Initial action noise standard deviation.
        end_sigma : float
            Final action noise standard deviation after annealing.
        anneal_timesteps : int
            Number of steps over which to linearly anneal noise.
        verbose : int
            Verbosity level.
        """
        super().__init__(verbose)
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)
        self.anneal_timesteps = max(1, int(anneal_timesteps))
        self.total_steps = 0

    def _on_step(self) -> bool:
        """
        Callback executed at every step.
        Updates the action noise according to linear schedule.

        Returns
        -------
        bool
            True to continue training.
        """
        self.total_steps += 1
        self._update_noise()
        return True

    def _update_noise(self):
        """
        Updates the model's action noise sigma based on current progress.
        Works with both NormalActionNoise and Ornstein-Uhlenbeck noise.
        """
        if not hasattr(self.model, "action_noise") or self.model.action_noise is None:
            return

        frac = min(1.0, float(self.total_steps) / float(self.anneal_timesteps))
        new_sigma = float(self.start_sigma + frac * (self.end_sigma - self.start_sigma))

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
    Build a Stable-Baselines3 model with algorithm-specific settings.

    Supports:
        - SAC: ent_coef (alpha), target_entropy scaling
        - TD3/DDPG: action noise (OU or Normal)
        - Linear learning rate schedule
        - Custom network architecture

    Parameters
    ----------
    algo_name : str
        Algorithm name (SAC, TD3, DDPG)
    vec_env : VecEnv
        Vectorized training environment
    config : dict
        Training configuration dictionary
    tb_log_name : str
        Path for TensorBoard logging

    Returns
    -------
    model : BaseAlgorithm
        Initialized SB3 model instance.
    """
    algo = algo_name.upper()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo: {algo}")

    ModelClass = ALGOS[algo]
    algo_cfg = config.get(algo.lower(), {}) or {}

    batch_size = int(algo_cfg.get("batch_size", 256))
    buffer_size = int(algo_cfg.get("buffer_size", 1_000_000))
    gamma = float(algo_cfg.get("gamma", 0.99))
    tau = float(algo_cfg.get("tau", 0.005))
    learning_starts = int(algo_cfg.get("learning_starts", 5000))
    lr_conf = algo_cfg.get("learning_rate", 1e-4)

    if bool(algo_cfg.get("lr_schedule", True)):
        try:
            lr = get_linear_fn(
                float(lr_conf),
                float(lr_conf) * 0.05,
                end_fraction=float(algo_cfg.get("lr_schedule_end_frac", 0.8))
            )
        except Exception:
            lr = float(lr_conf)
    else:
        lr = float(lr_conf)

    action_dim = None
    try:
        action_space = vec_env.action_space
    except Exception:
        action_space = None

    if isinstance(action_space, gym.spaces.Box):
        action_dim = int(np.prod(action_space.shape))

    action_noise = None
    if algo in ("DDPG", "TD3") and action_dim is not None:
        noise_sigma = float(algo_cfg.get("noise_sigma", 0.2))
        noise_type = str(algo_cfg.get("noise_type", "ou")).lower()
        if noise_type == "ou":
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim),
                sigma=noise_sigma * np.ones(action_dim)
            )
        else:
            action_noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=noise_sigma * np.ones(action_dim)
            )

    net_arch = algo_cfg.get("net_arch", [400, 300])
    if isinstance(net_arch, str):
        try:
            net_arch = eval(net_arch)
        except Exception:
            net_arch = [400, 300]

    policy_kwargs = {"net_arch": net_arch, "activation_fn": nn.ReLU, "normalize_images": False}

    extra_kwargs = {}
    if algo == "SAC":
        ent_conf = algo_cfg.get("ent_coef", "auto")
        if isinstance(ent_conf, str) and ent_conf.lower() == "auto":
            ent_coef = "auto"
        else:
            try:
                ent_coef = float(ent_conf)
            except Exception:
                ent_coef = "auto"
        extra_kwargs["ent_coef"] = ent_coef

        scale = algo_cfg.get("target_entropy_scale", None)
        if scale is not None and action_dim is not None:
            try:
                scale = float(scale)
                target_entropy = -float(action_dim) * scale
                extra_kwargs["target_entropy"] = target_entropy
            except Exception:
                pass

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
    if action_noise is not None:
        model_kwargs["action_noise"] = action_noise

    model_kwargs.update(extra_kwargs)
    model = ModelClass(**model_kwargs)
    return model


# --------------------------------------------------------------
#  MAIN TRAINING LOOP
# --------------------------------------------------------------
def main(args, algo_name):
    """
    Main training script.

    Loads config, builds envs and model, sets up logging/callbacks,
    and runs the learning loop.
    """
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
        vec_env = build_vec_envs(config)
        model = build_model(algo_name, vec_env, config, tb_log_name)
        checkpoint_dir = os.path.join(models_dir, algo_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{algo_name}_ckpt"
        )

        eval_env = build_vec_envs(config)
        try:
            if isinstance(vec_env, VecNormalize) and isinstance(eval_env, VecNormalize):
                eval_env.training = False
                eval_env.norm_reward = vec_env.norm_reward
                eval_env.obs_rms = vec_env.obs_rms
                eval_env.ret_rms = vec_env.ret_rms
        except Exception:
            pass
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

        ddpg_cfg = config.get("ddpg", {}) or {}
        start_sigma = float(ddpg_cfg.get("noise_sigma", 0.2))
        end_sigma = float(ddpg_cfg.get("noise_sigma_end", 0.02))
        anneal_timesteps = timesteps
        noise_cb = NoiseAnnealingCallback(start_sigma, end_sigma, anneal_timesteps)

        new_logger = configure(tb_log_name, ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        callback_list = [checkpoint_cb, episode_cb, step_cb, eval_callback, noise_cb]
        start = time.time()
        model.learn(total_timesteps=timesteps, callback=callback_list)
        duration_min = (time.time() - start) / 60.0
        print(f"Training completed in {duration_min:.2f} minutes")

        final_path = os.path.join(checkpoint_dir, f"{algo_name}_final")
        model.save(final_path)
        try:
            if isinstance(vec_env, VecNormalize):
                vnorm_path = os.path.join(checkpoint_dir, f"{algo_name}_vecnormalize.pkl")
                vec_env.save(vnorm_path)
                print(f"Saved VecNormalize to {vnorm_path}")
        except Exception:
            pass

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--algo", type=str, help="Optional: SAC, TD3, or DDPG")
    args = parser.parse_args()
    main(args, args.algo)
