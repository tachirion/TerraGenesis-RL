import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from envs import make_env_fn


def make_vec_env(seed, max_steps, stable_steps_required):
    return VecMonitor(
        DummyVecEnv([
            make_env_fn(
                max_steps=max_steps,
                stable_steps_required=stable_steps_required,
                seed=seed
            )
        ])
    )


class OptunaCallback(BaseCallback):
    """
    Custom callback for Optuna pruning based on mean reward.
    Reports intermediate rewards and raises TrialPruned if unpromising.
    """
    def __init__(self, trial, eval_env, n_eval_episodes=5, eval_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.last_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_step >= self.eval_freq:
            self.last_step = self.num_timesteps
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                warn=False
            )
            self.trial.report(mean_reward, self.num_timesteps)

            print(f"[Trial {self.trial.number}] Step {self.num_timesteps}: Mean reward = {mean_reward:.2f}")

            if self.trial.should_prune():
                print(f"[Trial {self.trial.number}] Trial pruned at step {self.num_timesteps}.")
                raise optuna.exceptions.TrialPruned()
        return True


def optimize_sac(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.02)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    net_size = trial.suggest_categorical("net_size", [128, 256, 400])

    policy_kwargs = dict(net_arch=[net_size, net_size])

    env = make_vec_env(seed=42, max_steps=200, stable_steps_required=5)
    eval_env = make_vec_env(seed=100, max_steps=200, stable_steps_required=5)  # separate eval env

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=300_000,
        learning_starts=2_000,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    pruning_callback = OptunaCallback(trial, eval_env, n_eval_episodes=5, eval_freq=10_000)

    try:
        model.learn(
            total_timesteps=80_000,
            progress_bar=False,
            callback=pruning_callback
        )
    except optuna.exceptions.TrialPruned:
        raise

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, warn=False)
    return mean_reward


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="sac_optimization",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20_000, interval_steps=10_000)
    )

    study.optimize(
        optimize_sac,
        n_trials=25,
        n_jobs=1,
        show_progress_bar=True
    )

    print("\n=== BEST PARAMETERS ===")
    print(study.best_params)
    print(f"Best reward: {study.best_value:.2f}")
