import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TerraGenesisEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps=200, stable_steps_required=5, seed=None):
        super().__init__()
        self.obs_dim = 5
        self.act_dim = 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
        self.max_steps = max_steps
        self.target = np.array([0.3, 0.5, 0.4, 0.6, 0.0], dtype=np.float32)
        self.state = None
        self.step_count = 0
        self.stable_counter = 0
        self.stable_steps_required = stable_steps_required
        self.prev_state = None
        self.resource_budget = 200.0
        self.resource_used = 0.0
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(-0.5, 0.5, size=(self.obs_dim,)).astype(np.float32)
        self.prev_state = self.state.copy()
        self.step_count = 0
        self.stable_counter = 0
        self.resource_used = 0.0
        return self.state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.prev_state = self.state.copy()
        delta = 0.05 * action[:4]
        noise = self.rng.normal(0, 0.005, size=(4,))
        self.state[:4] = np.clip(self.state[:4] + delta + noise, -1.0, 1.0)
        self.state[4] = np.clip(-np.sum(np.abs(self.state[:4] - self.target[:4]))/2.0, -1.0, 1.0)
        action_cost = float(np.sum(np.square(action)))
        self.resource_used += action_cost
        r_task = -float(np.sum((self.state[:4] - self.target[:4])**2))
        instability = float(np.sum(np.abs(self.state - self.prev_state)))
        r_stab = -5.0 * instability
        r_cost = -0.05 * action_cost
        reward = r_task + r_stab + r_cost
        reward = float(np.clip(reward, -100.0, 100.0))
        self.step_count += 1

        diff = float(np.linalg.norm(self.state[:4] - self.target[:4]))
        if diff < 0.05:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        terminated = False
        truncated = False
        if self.stable_counter >= self.stable_steps_required:
            terminated = True
            reward += 50.0
        if self.step_count >= self.max_steps:
            truncated = True
        if self.resource_used > self.resource_budget:
            terminated = True
            reward -= 50.0

        info = {
            "true_habitability": -float(np.sum((self.state[:4] - self.target[:4])**2)),
            "instability": instability,
            "resource_used": self.resource_used
        }

        return self.state, reward, terminated, truncated, info

def make_env_fn(max_steps=200, stable_steps_required=5, seed=None):
    def _init():
        env = TerraGenesisEnv(max_steps=max_steps, stable_steps_required=stable_steps_required, seed=seed)
        return env
    return _init
