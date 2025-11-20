import gymnasium as gym
from gymnasium import spaces
import numpy as np


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
        # 0 = Temperature
        # 1 = Oxygen
        # 2 = Pressure
        # 3 = Water
        # 4 = Biomass
        # 5 = Habitability (computed)

        self.obs_dim = 6
        self.act_dim = 5

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.act_dim,),
            dtype=np.float32
        )

        self.max_steps = max_steps
        self.step_count = 0
        self.stable_steps_required = stable_steps_required
        self.stable_counter = 0

        self.rng = np.random.default_rng(seed)

        self.target = np.array([
            0.2,   # Temperature
            0.5,   # Oxygen
            0.4,   # Pressure
            0.6,   # Water
            0.3    # Biomass
        ], dtype=np.float32)

        self.resource_budget = 300.0
        self.resource_used = 0.0

        self.state = None
        self.prev_state = None

    # ==============================================================
    # RESET
    # ==============================================================

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Initialize physical world
        self.state = self.rng.uniform(
            low=-0.2,
            high=0.2,
            size=(self.obs_dim,)
        ).astype(np.float32)

        self.state[5] = self.compute_habitability(self.state)

        self.prev_state = self.state.copy()
        self.step_count = 0
        self.stable_counter = 0
        self.resource_used = 0.0

        return self.state, {}

    # ==============================================================
    # COMPUTE HABITABILITY (0 = bad, 1 = perfect)
    # ==============================================================

    def compute_habitability(self, s):
        diff = np.abs(s[:5] - self.target).sum()
        return float(1.0 - diff / 5.0)

    # ==============================================================
    # RANDOM THREAT GENERATOR
    # ==============================================================

    def apply_random_event(self):
        """
        TerraGenesis-style disasters.
        Each step has a 10% chance of an event.
        """

        roll = self.rng.random()

        if roll > 0.10:     # 10% chance of event
            return "none"

        event_roll = self.rng.random()

        # ------------ Asteroid Impact -------------
        if event_roll < 0.2:
            self.state[2] -= 0.15     # Pressure drop
            self.state[3] -= 0.10     # Water flash vaporized
            return "asteroid"

        # ------------ Solar Flare ------------------
        elif event_roll < 0.4:
            self.state[0] += 0.12     # Heat spike
            return "solar_flare"

        # ------------ Volcanic Eruption ------------
        elif event_roll < 0.6:
            self.state[1] += 0.1      # Oxygen boost
            self.state[2] += 0.1      # Pressure change
            return "volcano"

        # ------------ Drought ----------------------
        elif event_roll < 0.8:
            self.state[3] -= 0.12
            return "drought"

        # ------------ Flood ------------------------
        else:
            self.state[3] += 0.12
            self.state[4] -= 0.1       # Biomass damage
            return "flood"

    # ==============================================================
    # STEP FUNCTION
    # ==============================================================

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.prev_state = self.state.copy()

        # === SIMULATION UPDATE ===
        delta = 0.05 * action[:4]
        noise = self.rng.normal(0, 0.005, size=(4,))
        self.state[:4] = np.clip(self.state[:4] + delta + noise, -1.0, 1.0)

        # global habitability indicator
        self.state[4] = np.clip(
            -np.sum(np.abs(self.state[:4] - self.target[:4])) / 2.0,
            -1.0,
            1.0
        )

        action_cost = float(np.sum(np.square(action)))
        self.resource_used += action_cost

        reward = 0.0

        # ------------------------------------------------------------------
        # 1) Weighted tracking penalty
        # ------------------------------------------------------------------
        errors = (self.state[:4] - self.target[:4]) ** 2
        weighted_error = np.sum(self.importance * errors)
        r_task = -weighted_error
        reward += r_task

        # ------------------------------------------------------------------
        # 2) Smoothness (reward stable control)
        # ------------------------------------------------------------------
        instability = float(np.sum(np.abs(self.state - self.prev_state)))
        r_smooth = 1.0 - np.tanh(instability)
        reward += r_smooth

        # ------------------------------------------------------------------
        # 3) Resource cost (small penalty)
        # ------------------------------------------------------------------
        r_cost = -0.05 * action_cost
        reward += r_cost

        # ------------------------------------------------------------------
        # 4) Bonuses for entering stable zones
        # ------------------------------------------------------------------
        for i in range(4):
            now_good = abs(self.state[i] - self.target[i]) < 0.05
            was_good = abs(self.prev_state[i] - self.target[i]) < 0.05

            if now_good and not was_good:
                reward += 5.0  # milestone
            elif was_good and not now_good:
                reward -= 7.0  # penalty

        # ------------------------------------------------------------------
        # 5) Bonus for global habitability
        # ------------------------------------------------------------------
        diff = float(np.linalg.norm(self.state[:4] - self.target[:4]))

        if diff < 0.05:
            self.stable_counter += 1
            reward += 20.0  # global stability reward
        else:
            self.stable_counter = 0

        # === TERMINATION RULES ===
        self.step_count += 1
        terminated = False
        truncated = False

        if self.stable_counter >= self.stable_steps_required:
            terminated = True
            reward += 100.0  # big win

        if self.step_count >= self.max_steps:
            truncated = True

        if self.resource_used > self.resource_budget:
            terminated = True
            reward -= 50.0

        info = {
            "weighted_error": weighted_error,
            "instability": instability,
            "resource_used": self.resource_used
        }

        return self.state, float(reward), terminated, truncated, info

# ==============================================================
# ENV FACTORY FOR VEC/ASYNC TRAINING
# ==============================================================

def make_env_fn(max_steps=300, stable_steps_required=10, seed=None):
    def _init():
        return TerraGenesisEnv(
            max_steps=max_steps,
            stable_steps_required=stable_steps_required,
            seed=seed
        )
    return _init
