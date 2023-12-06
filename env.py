import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Callable
import numpy as np
from gym.spaces import Box, Discrete, Tuple

RIDER_AWC_MIN = 0  # in watts/kg
RIDER_AWC_MAX = 10000  # in watts/kg

RIDER_POWER_MIN = 0  # in watts
RIDER_POWER_MAX = 3000  # in watts

RIDER_VELOCITY_MIN = -28  # in m/s
RIDER_VELOCITY_MAX = 28  # in m/s

RIDER_AWC_j = 9758  # in joules
RIDER_CP_w = 234  # in watts
RIDER_MASS_kg = 70  # in kilos

G = 9.81  # in m/s^2
ROLLING_RESISTANCE_COEFF = 0.0046


def max_power(AWC_j):
    return 7e-6 * AWC_j**2 + 0.0023 * AWC_j + RIDER_CP_w


class RiderEnv:
    def __init__(
        self,
        gradient: Callable[[float], float],
        distance: int = 1000,
        num_actions: int = 10,
        reward: Callable[[int], float] = None,
    ):
        if not callable(gradient):
            raise ValueError("Gradient must be a callable function")

        self.render_mode = None

        self.course_fn = gradient
        self.START_DISTANCE = 0
        self.COURSE_DISTANCE = distance

        # State/Action spaces
        # ---------------------------
        self.observation_space = Tuple(
            [
                Box(low=RIDER_POWER_MIN, high=RIDER_POWER_MAX, shape=(1,), dtype=float),
                Box(
                    low=RIDER_VELOCITY_MIN,
                    high=RIDER_VELOCITY_MAX,
                    shape=(1,),
                    dtype=float,
                ),
                Box(low=-100, high=100, shape=(1,), dtype=float),
                Box(low=0, high=1, shape=(1,), dtype=float),
                Box(low=RIDER_AWC_MIN, high=RIDER_AWC_MAX, shape=(1,), dtype=float),
            ]
        )
        self.action_space = Discrete(num_actions)
        self.reward = reward if reward else self.default_reward

        # Environment variables
        # ---------------------------
        self.step_count = 0

        # Agent variables
        # ---------------------------
        self.cur_AWC_j = RIDER_AWC_j
        self.cur_velocity = 0
        self.cur_position = self.START_DISTANCE

    @property
    def state(self) -> np.ndarray:
        return np.array(
            [
                max_power(self.cur_AWC_j),  # power_max_w
                self.cur_velocity,  # velocity
                self.course_fn(self.cur_position),  # gradient
                self.cur_position / self.COURSE_DISTANCE,  # percent_complete
                self.cur_AWC_j,  # AWC
            ]
        )

    def default_reward(self: int) -> float:
        reward = -1

        if self.cur_position >= self.COURSE_DISTANCE:
            reward += 100

        if self.cur_velocity <= 0:
            reward -= 100

        return reward

    def step(self, action: int):
        # Error handling
        # -------------------------------------------------
        if self.action_space.n < action < 0:
            raise ValueError("Invalid action")

        # Step count
        self.step_count += 1

        # Normalize action
        action = action / self.action_space.n

        # Physiological calculations
        # -------------------------------------------------

        power_max_w = max_power(self.cur_AWC_j)

        power_agent_w = action * power_max_w

        if power_agent_w > RIDER_CP_w:
            # Fatigue
            fatigue = -(power_agent_w - RIDER_CP_w)
        else:
            # Recovery
            p_agent_adj_w = 0.0879 * power_agent_w + 204.5
            fatigue = -(p_agent_adj_w - RIDER_CP_w)

        self.cur_AWC_j = min(self.cur_AWC_j + fatigue, RIDER_AWC_j)

        # Environment/Agent calculations
        # -------------------------------------------------

        # Resistance calculations
        # -----------------------

        slope_percent = self.course_fn(self.cur_position)
        slope_radians = np.arctan(slope_percent / 100)

        horizontal_gravity_force = G * RIDER_MASS_kg * np.sin(slope_radians)

        rolling_resistance_force = G * RIDER_MASS_kg * ROLLING_RESISTANCE_COEFF

        drag_force = 0.19 * (self.cur_velocity) ** 2

        total_resistance_w = (
            horizontal_gravity_force + rolling_resistance_force + drag_force
        ) * self.cur_velocity

        # Velocity calculations
        # ---------------------

        cur_KE_j = (1 / 2) * RIDER_MASS_kg * (self.cur_velocity) ** 2

        final_KE_j = cur_KE_j - total_resistance_w + power_agent_w

        self.cur_velocity = np.sign(final_KE_j) * np.sqrt(
            2 * abs(final_KE_j) / RIDER_MASS_kg
        )

        # Rider position calculations
        # ---------------------------

        new_position = self.cur_position + self.cur_velocity

        self.cur_position = min(new_position, self.COURSE_DISTANCE)

        # Rewards and termination
        # -----------------------

        terminated = 0
        truncated = 0

        if self.cur_position >= self.COURSE_DISTANCE:
            terminated = 1

        reward = self.reward(self.state)

        # State and info
        # -------------------------------------------------

        info = {
            "power_max_w": power_max_w,
            "velocity": self.cur_velocity,
            "gradient": slope_percent,
            "percent_complete": self.cur_position / self.COURSE_DISTANCE,
            "AWC": self.cur_AWC_j,
            "position": self.cur_position,
            "action": action,
            "power_output": power_agent_w,
            "reward": reward,
        }

        return self.state, reward, terminated, truncated, info

    def reset(self, start: int = None):
        self.step_count = 0
        self.cur_AWC_j = RIDER_AWC_j
        self.cur_velocity = 0
        self.cur_position = start if start else self.START_DISTANCE

        info = {
            "power_max_w": max_power(self.cur_AWC_j),
            "velocity": self.cur_velocity,
            "gradient": self.course_fn(self.cur_position),
            "percent_complete": self.cur_position / self.COURSE_DISTANCE,
            "AWC": self.cur_AWC_j,
            "position": self.cur_position,
            "action": None,
            "power_output": None,
            "reward": None,
        }

        return self.state, info
