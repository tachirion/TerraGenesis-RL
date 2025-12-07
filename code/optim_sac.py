import os
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from envs import make_env_fn

# -----------------------------
# Vectorized environment helper
# -----------------------------
def make_vec_env(seed, max_steps=300, stable_steps_required=10):
    return VecMonitor(
        DummyVecEnv([
            make_env_fn(
                max_steps=max_steps,
                stable_steps_required=stable_steps_required,
                seed=seed
            )
        ])
    )

# -----------------------------
# Optuna pruning callback
# -----------------------------
class OptunaCallback(BaseCallback):
    """Reports mean reward to Optuna and prunes unpromising trials."""
    def __init__(self, trial, eval_env, n_eval_episodes=5, eval_freq=50_000, verbose=1):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.last_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_step >= self.eval_freq:
            self.last_step = self.num_timesteps
            # explicitly render=False to avoid any GUI calls
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                warn=False,
                render=False
            )
            self.trial.report(mean_reward, self.num_timesteps)
            print(f"[Trial {self.trial.number}] Step {self.num_timesteps}: Mean reward = {mean_reward:.2f}")

            if self.trial.should_prune():
                print(f"[Trial {self.trial.number}] Trial pruned at step {self.num_timesteps}.")
                raise optuna.exceptions.TrialPruned()
        return True

# -----------------------------
# SAC hyperparameter optimization
# -----------------------------
def optimize_sac(trial):
    # ----- Lowered/stable hyperparameter search space -----
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.97)
    tau = trial.suggest_float("tau", 0.0003, 0.002)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    net_size = trial.suggest_categorical("net_size", [256, 400, 512])
    alpha = trial.suggest_float("alpha", 1e-5, 0.05, log=True)  # entropy coefficient
    target_entropy_scale = trial.suggest_float("target_entropy_scale", 0.5, 0.8)

    policy_kwargs = dict(net_arch=[net_size, net_size])

    env = make_vec_env(seed=42, max_steps=300, stable_steps_required=10)
    eval_env = make_vec_env(seed=100, max_steps=300, stable_steps_required=10)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=1_000_000,
        learning_starts=20_000,
        train_freq=1,
        gradient_steps=1,
        ent_coef=alpha,
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    if model.ent_coef_optimizer is not None:
        model.target_entropy = target_entropy_scale * -env.action_space.shape[0]

    # Checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=f"./checkpoints/trial_{trial.number}",
        name_prefix="sac"
    )

    pruning_cb = OptunaCallback(trial, eval_env, n_eval_episodes=5, eval_freq=50_000)

    try:
        model.learn(
            total_timesteps=250_000,
            callback=[pruning_cb, checkpoint_cb],
            progress_bar=True
        )
    except optuna.exceptions.TrialPruned:
        raise

    # Final evaluation, headless
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False, warn=False)
    return mean_reward

# -----------------------------
# Main Optuna study
# -----------------------------
if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)

    study = optuna.create_study(
        study_name="sac_terra_stable",
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=50_000,
            max_resource=250_000,
            reduction_factor=3
        )
    )

    study.optimize(
        optimize_sac,
        n_trials=30,
        n_jobs=1,
        show_progress_bar=True
    )

    print("\n=== BEST PARAMETERS ===")
    print(study.best_params)
    print(f"Best reward: {study.best_value:.2f}")
