import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import cast


class TerraGenesisEnv(gym.Env):
    """
    TerraGenesis-inspired RL environment.
    Agent must keep planetary conditions within habitable ranges
    while random threats occur (asteroid, solar flare, volcanic event, etc.)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=300, stable_steps_required=10, seed=None):
        super().__init__()

        # --------- State Dimensions --------------
        # 0 = Temperature, 1 = Oxygen, 2 = Pressure, 3 = Water, 4 = Biomass, 5 = Habitability
        self.obs_dim = 6
        self.act_dim = 5

        self.observation_space = spaces.Box(
            low=np.full(self.obs_dim, -1.0, dtype=np.float32),
            high=np.full(self.obs_dim, 1.0, dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.full(self.act_dim, -1.0, dtype=np.float32),
            high=np.full(self.act_dim, 1.0, dtype=np.float32),
            dtype=np.float32
        )

        self.max_steps = max_steps
        self.step_count = 0
        self.stable_steps_required = stable_steps_required
        self.stable_counter = 0

        self.rng = np.random.default_rng(seed)

        self.target = np.array([0.2, 0.5, 0.4, 0.6, 0.3], dtype=np.float32)
        self.resource_budget = 300.0
        self.resource_used = 0.0

        self.state = None
        self.prev_state = None
        self.importance = np.ones(4, dtype=np.float32)

    # ----------------------------------------------------
    # RESET
    # ----------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(low=-0.2, high=0.2, size=(self.obs_dim,)).astype(np.float32)
        self.state[4] = np.clip(self.state[4], -1.0, 1.0)
        self.state[5] = self.compute_habitability(self.state)
        self.prev_state = self.state.copy()
        self.step_count = 0
        self.stable_counter = 0
        self.resource_used = 0.0
        return self.state, {}

    # ----------------------------------------------------
    # HABITABILITY
    # ----------------------------------------------------
    def compute_habitability(self, s):
        diff = np.abs(s[:5] - self.target).sum()
        return float(1.0 - diff / 5.0)

    # ----------------------------------------------------
    # RANDOM EVENTS
    # ----------------------------------------------------
    def apply_random_event(self):
        roll = self.rng.random()
        if roll > 0.10:
            return "none"
        event_roll = self.rng.random()
        if event_roll < 0.2:
            self.state[2] -= 0.15
            self.state[3] -= 0.10
            self.state[4] -= 0.05
            event = "asteroid"
        elif event_roll < 0.4:
            self.state[0] += 0.12
            event = "solar_flare"
        elif event_roll < 0.6:
            self.state[1] += 0.10
            self.state[2] += 0.10
            event = "volcano"
        elif event_roll < 0.8:
            self.state[3] -= 0.12
            self.state[4] -= 0.06
            event = "drought"
        else:
            self.state[3] += 0.12
            self.state[4] -= 0.10
            event = "flood"

        self.state[:5] = np.clip(self.state[:5], -1.0, 1.0)
        return event

    # ----------------------------------------------------
    # STEP
    # ----------------------------------------------------
    def step(self, action):
        action = np.clip(action, cast(spaces.Box, self.action_space).low, cast(spaces.Box, self.action_space).high)
        self.prev_state = self.state.copy()

        delta = 0.05 * action[:4]
        noise = self.rng.normal(0, 0.005, size=(4,))
        self.state[:4] = np.clip(self.state[:4] + delta + noise, -1.0, 1.0)

        biomass_delta = 0.03 * action[4] + self.rng.normal(0, 0.01)
        self.state[4] = float(np.clip(self.state[4] + biomass_delta, -1.0, 1.0))

        event = self.apply_random_event()
        self.state[5] = self.compute_habitability(self.state)
        self.state = np.clip(self.state, -1.0, 1.0).astype(np.float32)

        action_cost = float(np.sum(np.square(action)))
        self.resource_used += action_cost

        # --- Reward computation ---
        reward = 0.0
        errors = (self.state[:4] - self.target[:4]) ** 2
        weighted_error = np.sum(self.importance * errors)
        reward += -weighted_error
        instability = float(np.sum(np.abs(self.state - self.prev_state)))
        reward += 1.0 - np.tanh(instability)
        reward += -0.05 * action_cost

        for i in range(4):
            now_good = abs(self.state[i] - self.target[i]) < 0.05
            was_good = abs(self.prev_state[i] - self.target[i]) < 0.05
            if now_good and not was_good:
                reward += 5.0
            elif was_good and not now_good:
                reward -= 7.0

        diff = float(np.linalg.norm(self.state[:4] - self.target[:4]))
        if diff < 0.05:
            self.stable_counter += 1
            reward += 20.0
        else:
            self.stable_counter = 0

        self.step_count += 1
        terminated = False
        truncated = False
        if self.stable_counter >= self.stable_steps_required:
            terminated = True
            reward += 100.0
        if self.step_count >= self.max_steps:
            truncated = True
        if self.resource_used > self.resource_budget:
            terminated = True
            reward -= 50.0

        # --- Threats info ---
        threats = []
        labels = ["Temperature", "Oxygen", "Pressure", "Water"]
        for i, val in enumerate(self.state[:4]):
            if val < -0.6:
                threats.append({
                    "id": f"{i}_low",
                    "type": labels[i],
                    "severity": "High",
                    "description": f"{labels[i]} dangerously low!",
                    "icon": "⚠️"
                })
            elif val > 0.8:
                threats.append({
                    "id": f"{i}_high",
                    "type": labels[i],
                    "severity": "Medium",
                    "description": f"{labels[i]} too high!",
                    "icon": "🔥"
                })

        info = {
            "weighted_error": weighted_error,
            "threats": threats,
            "event": event,
            "true_habitability": float(self.state[5]),
            "instability": float(instability),
            "resource_used": float(self.resource_used),
        }

        # --- Episode metrics for SB3 / VecMonitor ---
        if terminated or truncated:
            # keep the episode dict as well (SB3 expects 'r' etc), but since Monitor may
            # overwrite/replace it, also expose metrics at top level (above)
            info["episode"] = {
                "r": float(reward),
                "true_habitability": float(self.state[5]),
                "instability": float(instability),
                "resource_used": float(self.resource_used),
            }

        return self.state, float(reward), terminated, truncated, info

    # ----------------------------------------------------
    # RENDER
    # ----------------------------------------------------
    def render(self, mode="human"):
        if mode == "human":
            print("Step:", self.step_count, "State:", np.round(self.state, 3))
            return None
        else:
            return self.state

    def close(self):
        pass


# ----------------------------------------------------
# ENV FACTORY
# ----------------------------------------------------
def make_env_fn(max_steps=300, stable_steps_required=10, seed=None):
    def _init():
        return TerraGenesisEnv(max_steps=max_steps, stable_steps_required=stable_steps_required, seed=seed)
    return _init
