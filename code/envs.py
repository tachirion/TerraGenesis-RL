# envs.py (corrected)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict


class TerraGenesisEnv(gym.Env):
    """
    TerraGenesis-inspired RL environment (corrected for stable RL training).

    Key changes from original:
    - Smaller per-step action effects and softened random events
    - Reduced/stabilized reward magnitudes (no huge spikes)
    - Clipped reward to a reasonable range for easier normalization
    - Consistent dtype and seeding
    - Configurable constants at top for easy tuning
    """

    metadata = {"render_modes": ["human"]}

    # Tunable constants
    ACTION_SCALE = 0.03          # effect of continuous actions on first 4 dims (was 0.05)
    BIOMASS_SCALE = 0.02         # effect of biomass action (was 0.03)
    RANDOM_EVENT_SCALE = 0.8     # soften disaster magnitudes (was 1.0)
    STABILITY_BONUS = 2.0        # smaller terminal/stability bonus (was 20.0)
    STABILITY_THRESHOLD = 0.05   # threshold for being considered 'close' to target
    ACTION_COST_WEIGHT = 0.01    # weight on action cost (was 0.05)
    ERROR_WEIGHT = 0.5           # weight for -error term (was implicitly 1)
    INSTABILITY_WEIGHT = 0.5     # weight for instability reward component
    REWARD_CLIP = 5.0            # clip final reward to [-REWARD_CLIP, REWARD_CLIP]

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

        self.max_steps = int(max_steps)
        self.step_count = 0
        self.stable_steps_required = int(stable_steps_required)
        self.stable_counter = 0

        self.rng = np.random.default_rng(seed)

        # target for first 5 dims
        self.target = np.array([0.2, 0.5, 0.4, 0.6, 0.3], dtype=np.float32)
        self.resource_budget = 300.0
        self.resource_used = 0.0

        self.state = None
        self.prev_state = None
        # importance weights used in reward for first 4 dims
        self.importance = np.ones(4, dtype=np.float32)

        # human-readable mappings for decoded actions (unchanged)
        self._action_text = {
            "Temperature": {
                "strong_decrease": "Deploy space mirrors to cool surface",
                "decrease": "Reduce greenhouse generators",
                "maintain": "Hold thermal systems stable",
                "increase": "Activate greenhouse factories",
                "strong_increase": "Release strong greenhouse gases"
            },
            "Oxygen": {
                "strong_decrease": "Scrub oxygen aggressively",
                "decrease": "Reduce photosynthesis systems",
                "maintain": "Maintain O2 balance",
                "increase": "Release engineered algae",
                "strong_increase": "Deploy oxygen factories"
            },
            "Pressure": {
                "strong_decrease": "Open vents to release atmosphere",
                "decrease": "Reduce atmospheric compressors",
                "maintain": "Maintain pressure systems",
                "increase": "Activate pressure generators",
                "strong_increase": "Inject heavy gases"
            },
            "Water": {
                "strong_decrease": "Force evaporation protocols",
                "decrease": "Reduce water pumps",
                "maintain": "Maintain water cycle",
                "increase": "Seed rain clouds",
                "strong_increase": "Release polar melt operations"
            },
            "Biomass": {
                "strong_decrease": "Cull invasive species",
                "decrease": "Reduce biomass growth programs",
                "maintain": "Maintain ecological balance",
                "increase": "Plant new vegetation",
                "strong_increase": "Deploy mass biome expansion"
            }
        }

        self._labels = ["Temperature", "Oxygen", "Pressure", "Water", "Biomass"]

        # tracking cumulative episode return for proper logging
        self.episode_return = 0.0

    # ----------------------------------------------------
    # RESET
    # ----------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # initialize small random state centered near zero
        self.state = self.rng.uniform(low=-0.2, high=0.2, size=(self.obs_dim,)).astype(np.float32)
        # ensure biomass and habitability within clip
        self.state[4] = float(np.clip(self.state[4], -1.0, 1.0))
        self.state[5] = float(self.compute_habitability(self.state))
        self.prev_state = self.state.copy()
        self.step_count = 0
        self.stable_counter = 0
        self.resource_used = 0.0
        self.episode_return = 0.0
        return self.state.astype(np.float32), {}

    # ----------------------------------------------------
    # HABITABILITY
    # ----------------------------------------------------
    def compute_habitability(self, s: np.ndarray) -> float:
        # smaller normalization: sum absolute diff across first-5, scaled so full match -> 1.0, large mismatch -> negative small
        diff = np.abs(s[:5] - self.target).sum()
        return float(1.0 - diff / 5.0)

    # ----------------------------------------------------
    # RANDOM EVENTS
    # ----------------------------------------------------
    def apply_random_event(self) -> str:
        roll = self.rng.random()
        if roll > 0.10:
            return "none"
        event_roll = self.rng.random()
        s = self.RANDOM_EVENT_SCALE
        if event_roll < 0.2:
            self.state[2] -= 0.15 * s
            self.state[3] -= 0.10 * s
            self.state[4] -= 0.05 * s
            event = "asteroid"
        elif event_roll < 0.4:
            self.state[0] += 0.12 * s
            event = "solar_flare"
        elif event_roll < 0.6:
            self.state[1] += 0.10 * s
            self.state[2] += 0.10 * s
            event = "volcano"
        elif event_roll < 0.8:
            self.state[3] -= 0.12 * s
            self.state[4] -= 0.06 * s
            event = "drought"
        else:
            self.state[3] += 0.12 * s
            self.state[4] -= 0.10 * s
            event = "flood"

        self.state[:5] = np.clip(self.state[:5], -1.0, 1.0)
        return event

    # ----------------------------------------------------
    # ACTION DECODING HELPERS
    # ----------------------------------------------------
    def interpret_action_value(self, value: float) -> str:
        if value <= -0.6:
            return "strong_decrease"
        elif value <= -0.2:
            return "decrease"
        elif value < 0.2:
            return "maintain"
        elif value < 0.6:
            return "increase"
        else:
            return "strong_increase"

    def decode_actions(self, action: np.ndarray) -> List[Dict]:
        decoded = []
        for i, dim in enumerate(self._labels):
            val = float(action[i])
            label = self.interpret_action_value(val)
            desc = self._action_text[dim][label]
            decoded.append({
                "dimension": dim,
                "value": val,
                "label": label,
                "description": desc
            })
        return decoded

    # ----------------------------------------------------
    # STEP
    # ----------------------------------------------------
    def step(self, action):
        # clamp actions
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        # decode actions
        action_raw = action.astype(float).tolist()
        action_decoded = self.decode_actions(action)
        action_decoded_str = ";".join([f"{d['dimension']}:{d['label']}" for d in action_decoded])

        # apply continuous effects (smaller, less noisy)
        delta = self.ACTION_SCALE * action[:4]
        noise = self.rng.normal(0, 0.004, size=(4,)).astype(np.float32)
        self.state[:4] = np.clip(self.state[:4] + delta + noise, -1.0, 1.0)

        # biomass update (softer)
        biomass_delta = self.BIOMASS_SCALE * action[4] + self.rng.normal(0, 0.008)
        self.state[4] = float(np.clip(self.state[4] + biomass_delta, -1.0, 1.0))

        # random disaster (softened)
        event = self.apply_random_event()

        # update habitability
        self.state[5] = float(self.compute_habitability(self.state))
        self.state = np.clip(self.state, -1.0, 1.0).astype(np.float32)

        # action cost (small)
        action_cost = float(np.sum(np.square(action)))
        self.resource_used += action_cost

        # ---- reward shaping (balanced & clipped) ----
        errors = (self.state[:4] - self.target[:4]) ** 2
        weighted_error = np.sum(self.importance * errors)

        instability = float(np.sum(np.abs(delta)))

        # Compose reward from terms with moderate weights
        r_error = -self.ERROR_WEIGHT * weighted_error
        r_instability = self.INSTABILITY_WEIGHT * (1.0 - float(np.tanh(instability)))
        r_action_cost = -self.ACTION_COST_WEIGHT * action_cost

        reward = r_error + r_instability + r_action_cost

        # small closeness bonus each step when really close
        diff = float(np.linalg.norm(self.state[:4] - self.target[:4]))
        if diff < self.STABILITY_THRESHOLD:
            self.stable_counter += 1
            # incremental small bonus per step close to target (not a huge spike)
            reward += float(self.STABILITY_BONUS * 0.25)
        else:
            self.stable_counter = 0

        # clip reward to reasonable range for stability (helps normalization)
        reward = float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))

        # accumulate episode return
        self.episode_return += reward

        # step bookkeeping
        self.step_count += 1
        terminated = (self.stable_counter >= self.stable_steps_required) or (self.resource_used > self.resource_budget)
        truncated = self.step_count >= self.max_steps

        # info dict
        info = {
            "step": self.step_count,
            "action_raw": action_raw,
            "action_decoded_str": action_decoded_str,
            "action_cost": action_cost,
            "reward": float(reward),
            "true_habitability": float(self.state[5]),
            "instability": float(instability),
            "resource_used": float(self.resource_used),
            "event_disaster": event if event != "none" else ""
        }

        # if episode ended, include episode summary with cumulative return
        if terminated or truncated:
            info["episode"] = {
                "r": float(self.episode_return),
                "true_habitability": float(self.state[5]),
                "instability": float(instability),
                "resource_used": float(self.resource_used),
            }

        return self.state.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

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
