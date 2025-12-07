"""
TerraGenesis RL Environment

Provides a Stable-Baselines3-compatible RL environment inspired by TerraGenesis.

Key Features:
 - 6-dimensional state: Temperature, Oxygen, Pressure, Water, Biomass, Habitability
 - 5-dimensional continuous action space with interpretable labels
 - Softened random events and smaller per-step action effects for stable training
 - Balanced reward shaping with error, instability, action cost, and stability bonuses
 - Optional seeding for reproducibility
 - Utility methods for decoding actions and computing habitability

Intended Usage:
 1. Create environment via TerraGenesisEnv() or make_env_fn()
 2. Use with any SB3 algorithm (SAC, TD3, DDPG)
 3. Supports vectorized environments and logging
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict


class TerraGenesisEnv(gym.Env):
    """
    A TerraGenesis-inspired reinforcement learning environment designed specifically
    for stable and well-behaved continuous control training.

    This environment simulates planetary terraforming where an agent attempts to
    stabilize temperature, oxygen, pressure, water, and biomass variables to reach
    Earth-like conditions.

    The environment produces stable RL training by:
    - Reducing the magnitude of action effects
    - Softening random environmental disaster events
    - Stabilizing reward magnitude and enforcing clipping
    - Avoiding discontinuities and overly large gradients
    - Normalizing observations to [-1, 1]
    - Ensuring deterministic seeding and reproducibility

    ----------------------------------------------------
    STATE SPACE (6-dimensional Box)
    ----------------------------------------------------
    Index | Name         | Meaning
    ------|--------------|---------------------------------------------------------
      0   | Temperature  | Scaled, normalized value in [-1, 1]
      1   | Oxygen       | Scaled, normalized value in [-1, 1]
      2   | Pressure     | Scaled, normalized value in [-1, 1]
      3   | Water        | Scaled, normalized value in [-1, 1]
      4   | Biomass      | Scaled, normalized value in [-1, 1]
      5   | Habitability | Derived score ∈ [-1, 1], computed from distance to target

    ----------------------------------------------------
    ACTION SPACE (5-dimensional Box)
    ----------------------------------------------------
    - Continuous values in [-1, 1]
    - Affects the first five state dimensions:
        action[0] = Temperature change command
        action[1] = Oxygen change command
        action[2] = Pressure change command
        action[3] = Water change command
        action[4] = Biomass growth/harm command

    ----------------------------------------------------
    TERMINATION CONDITIONS
    ----------------------------------------------------
    - The agent holds stable conditions near the target for N consecutive steps
    - OR the resource budget is exceeded
    - OR the maximum step limit is reached (truncation)

    ----------------------------------------------------
    REWARD FUNCTION
    ----------------------------------------------------
    reward = (
        - ERROR_WEIGHT * weighted_error
        + INSTABILITY_WEIGHT * (1 - tanh(instability))
        - ACTION_COST_WEIGHT * action_cost
        + stability_bonus_if_close
    )

    All rewards are clipped into the range [-REWARD_CLIP, REWARD_CLIP]
    to prevent extreme values during training.

    ----------------------------------------------------
    KEY CONSTANTS (tunable)
    ----------------------------------------------------
    ACTION_SCALE, BIOMASS_SCALE: control action → state effect magnitude
    RANDOM_EVENT_SCALE: controls magnitude of random disasters
    STABILITY_BONUS: small per-step bonus for being near target
    STABILITY_THRESHOLD: norm distance threshold to count as “stable”
    ACTION_COST_WEIGHT: small penalty for active intervention
    ERROR_WEIGHT: weight on squared error from target
    INSTABILITY_WEIGHT: reward for low volatility
    REWARD_CLIP: final clipping range

    ----------------------------------------------------
    DESIGN GOALS
    ----------------------------------------------------
    - Smooth reward landscape
    - No extremely spiky events
    - Predictable, stable gradients for policy optimization
    - Human-readable action decoding for interpretability
    - Deterministic behavior under seeding
    """

    metadata = {"render_modes": ["human"]}

    ACTION_SCALE = 0.03
    BIOMASS_SCALE = 0.02
    RANDOM_EVENT_SCALE = 0.8
    STABILITY_BONUS = 2.0
    STABILITY_THRESHOLD = 0.05
    ACTION_COST_WEIGHT = 0.01
    ERROR_WEIGHT = 0.5
    INSTABILITY_WEIGHT = 0.5
    REWARD_CLIP = 5.0

    def __init__(self, max_steps=300, stable_steps_required=10, seed=None):
        """
        Initialize the TerraGenesis environment.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps before truncation.
        stable_steps_required : int
            Number of consecutive steps near target required for termination.
        seed : int or None
            Random seed for deterministic runs.
        """
        super().__init__()

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
        self.target = np.array([0.2, 0.5, 0.4, 0.6, 0.3], dtype=np.float32)
        self.resource_budget = 300.0
        self.resource_used = 0.0
        self.state = None
        self.prev_state = None
        self.importance = np.ones(4, dtype=np.float32)
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
        self.episode_return = 0.0

    # ----------------------------------------------------
    # RESET
    # ----------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial random state.

        Parameters
        ----------
        seed : int or None
            Optional seed to reinitialize RNG.
        options : dict or None
            Unused, included for API compatibility.

        Returns
        -------
        observation : np.ndarray (shape = [6], dtype=float32)
            Initial normalized state vector.
        info : dict
            Empty dictionary for compatibility.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.rng.uniform(low=-0.2, high=0.2, size=(self.obs_dim,)).astype(np.float32)
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
        """
        Compute the planet's habitability score based on distance from the target
        across the first five dimensions.

        Habitability = 1 - (sum(|state[i] - target[i]|) / 5)

        Returns values in approximately [-1, 1], where:
        - 1.0 = perfect match with target
        - lower scores = worse conditions

        Parameters
        ----------
        s : np.ndarray
            State vector.

        Returns
        -------
        float
            Habitability score.
        """
        diff = np.abs(s[:5] - self.target).sum()
        return float(1.0 - diff / 5.0)

    # ----------------------------------------------------
    # RANDOM EVENTS
    # ----------------------------------------------------
    def apply_random_event(self) -> str:
        """
        Randomly applies a soft environmental disaster to the state with 10% probability.

        Events include:
        - asteroid impact
        - solar flare
        - volcanic eruption
        - drought
        - flood

        Effects are scaled down using RANDOM_EVENT_SCALE to avoid destabilizing
        training with large jumps.

        Returns
        -------
        str
            Name of event applied ("none" if no event occurred).
        """
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
    # ACTION DECODING
    # ----------------------------------------------------
    def interpret_action_value(self, value: float) -> str:
        """
        Convert a continuous action value into a discrete label describing
        the agent's qualitative intent.

        Ranges:
            [-1.0, -0.6] → "strong_decrease"
            (-0.6, -0.2] → "decrease"
            (-0.2, 0.2) → "maintain"
            [0.2, 0.6) → "increase"
            [0.6, 1.0] → "strong_increase"
        """
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
        """
        Convert each action dimension into a human-readable dictionary including:
        - variable name
        - raw value
        - qualitative label
        - natural-language description
        """
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
    # STEP FUNCTION
    # ----------------------------------------------------
    def step(self, action):
        """
        Apply an action, update the environment state, compute reward,
        detect termination, and return results.

        This method performs all core simulation logic including:
        - Clamping and applying action effects
        - Adding Gaussian noise
        - Updating biomass separately
        - Applying random environmental disasters
        - Computing habitability
        - Computing reward
        - Tracking resource usage
        - Determining episode termination/truncation

        Parameters
        ----------
        action : np.ndarray
            Continuous action vector of shape (5,).

        Returns
        -------
        observation : np.ndarray
            Updated state vector.
        reward : float
            Current step reward (clipped).
        terminated : bool
            True if the episode ends due to success or resource exhaustion.
        truncated : bool
            True if the episode ends due to step limit.
        info : dict
            Rich metadata about the step, including:
            - action interpretation
            - instability
            - disaster event
            - true habitability
            - resource usage
            - per-episode summary (if applicable)
        """
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        action_raw = action.astype(float).tolist()
        action_decoded = self.decode_actions(action)
        action_decoded_str = ";".join([f"{d['dimension']}:{d['label']}" for d in action_decoded])

        # ----------------------------------------------------
        # Apply continuous action effects
        # ----------------------------------------------------
        delta = self.ACTION_SCALE * action[:4]  # scaled effect
        noise = self.rng.normal(0, 0.004, size=(4,))  # small Gaussian noise
        self.state[:4] = np.clip(self.state[:4] + delta + noise, -1.0, 1.0)
        biomass_delta = self.BIOMASS_SCALE * action[4] + self.rng.normal(0, 0.008)  # evolves separately
        self.state[4] = float(np.clip(self.state[4] + biomass_delta, -1.0, 1.0))

        event = self.apply_random_event()

        self.state[5] = float(self.compute_habitability(self.state))
        self.state = np.clip(self.state, -1.0, 1.0).astype(np.float32)

        # ----------------------------------------------------
        # Resource usage penalty
        # ----------------------------------------------------
        action_cost = float(np.sum(np.square(action)))  # quadratic cost
        self.resource_used += action_cost

        # ----------------------------------------------------
        # Reward computation
        # ----------------------------------------------------
        errors = (self.state[:4] - self.target[:4]) ** 2
        weighted_error = np.sum(self.importance * errors)
        instability = float(np.sum(np.abs(delta)))  # magnitude of changes

        r_error = -self.ERROR_WEIGHT * weighted_error
        r_instability = self.INSTABILITY_WEIGHT * (1.0 - float(np.tanh(instability)))
        r_action_cost = -self.ACTION_COST_WEIGHT * action_cost
        reward = r_error + r_instability + r_action_cost

        # stability bonus if close to target
        diff = float(np.linalg.norm(self.state[:4] - self.target[:4]))
        if diff < self.STABILITY_THRESHOLD:
            self.stable_counter += 1
            reward += float(self.STABILITY_BONUS * 0.25)
        else:
            self.stable_counter = 0

        reward = float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))
        self.episode_return += reward

        # ----------------------------------------------------
        # Termination / truncation
        # ----------------------------------------------------
        self.step_count += 1
        terminated = (
            self.stable_counter >= self.stable_steps_required
            or self.resource_used > self.resource_budget
        )
        truncated = self.step_count >= self.max_steps

        # ----------------------------------------------------
        # Info dictionary
        # ----------------------------------------------------
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
        """
        Render the current state.
        Only a simple text output is provided.
        """
        if mode == "human":
            print("Step:", self.step_count, "State:", np.round(self.state, 3))
            return None
        else:
            return self.state

    def close(self):
        """
        Close any environment resources.
        Currently, no-op, included for API conformity.
        """
        pass


# ----------------------------------------------------
# ENV FACTORY (used for vectorized environments)
# ----------------------------------------------------
def make_env_fn(max_steps=300, stable_steps_required=10, seed=None):
    """
    Returns a function that creates a new TerraGenesisEnv instance.

    Parameters
    ----------
    max_steps : int
        Maximum episode length before truncation.
    stable_steps_required : int
        Steps close to target required for termination.
    seed : int or None
        Seed for environment RNG.
    """
    def _init():
        return TerraGenesisEnv(
            max_steps=max_steps,
            stable_steps_required=stable_steps_required,
            seed=seed
        )
    return _init
