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
    ):
        self.render_mode = None

        self.course_fn = gradient
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

        # Environment variables
        # ---------------------------
        self.time_step = 0

        # Agent variables
        # ---------------------------
        self.cur_AWC_j = RIDER_AWC_j
        self.cur_velocity = 0
        self.cur_position = self.START_DISTANCE

    def step(self, action: int):
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
        reward = -1

        if self.cur_position >= self.COURSE_DISTANCE:
            reward += 100
            terminated = 1

        if self.cur_velocity <= 0:
            reward -= 100

        # Observation and info
        # -------------------------------------------------

        observation = [
            power_max_w,
            self.cur_velocity,
            slope_percent,
            self.cur_position / self.COURSE_DISTANCE,
            self.cur_AWC_j,
        ]

        info = {
            **observation,
            "position": self.cur_position,
            "action": action,
            "power_output": power_agent_w,
            "reward": reward,
        }

        self.time_step += 1

        truncated = 0
        return observation, reward, terminated, truncated, info

    def reset(self, start: int = None):
        self.cur_AWC_j = RIDER_AWC_j
        self.cur_velocity = 0
        self.cur_position = start if start else self.START_DISTANCE

        observation = {
            "power_max_w": max_power(self.cur_AWC_j),
            "velocity": self.cur_velocity,
            "gradient": self.course_fn(self.cur_position),
            "percent_complete": self.cur_position / self.COURSE_DISTANCE,
            "AWC": self.cur_AWC_j,
        }

        info = {
            **observation,
            "position": self.cur_position,
            "action": None,
            "power_output": None,
            "reward": None,
        }

        return observation, info
