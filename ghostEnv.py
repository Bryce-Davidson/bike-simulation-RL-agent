import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Callable
import numpy as np
from env import RiderEnv

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


class GhostEnv:
    def __init__(
        self,
        ghost: Callable[[np.ndarray], int],
        reward: Callable[[int], float],
        gradient: Callable[[float], float],
        distance: int = 1000,
        num_actions: int = 10,
    ):
        # Error handling
        # ---------------------------
        if not callable(gradient):
            raise ValueError("Gradient must be a callable function")

        if not callable(reward):
            raise ValueError("Reward must be a callable function")

        if not callable(ghost):
            raise ValueError("Ghost strategy must be a callable function")

        # Environment variables
        # ---------------------------
        self.render_mode = None
        self.ghost = ghost
        self.ghost_env = RiderEnv(
            reward=lambda x: 0,
            gradient=gradient,
            distance=distance,
            num_actions=num_actions,
        )
        self.ghost_action = self.ghost(self.ghost_env.state)

        self.step_count = 0
        self.gradient = gradient
        self.START_DISTANCE = 0
        self.COURSE_DISTANCE = distance

        # State/Action spaces
        # ---------------------------
        self.action_space = num_actions
        self.reward = reward

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
                self.gradient(self.cur_position),  # gradient
                self.cur_position / self.COURSE_DISTANCE,  # percent_complete
                self.cur_AWC_j,  # AWC
                self.ghost_env.state[0],  # ghost_power_max_w
                self.ghost_env.state[1],  # ghost_velocity
                self.ghost_env.state[2],  # ghost_gradient
                self.ghost_env.state[3],  # ghost_percent_complete
                self.ghost_env.state[4],  # ghost_AWC
            ]
        )

    def info(self, meta: dict = {}) -> dict:
        state = self.state
        return {
            "power_max_w": state[0],
            "velocity": state[1],
            "gradient": state[2],
            "percent_complete": state[3],
            "AWC": state[4],
            "ghost_percent_complete": state[8],
            **meta,
        }

    def step(self, action: int):
        # Error handling
        # -------------------------------------------------
        if self.action_space < action < 0:
            raise ValueError(
                f"Invalid action: {action}. Must be between 0 and {self.action_space}"
            )
        # Ghost step
        # -------------------------------------------------
        self.ghost_env.step(self.ghost_action)
        self.ghost_action = self.ghost(self.ghost_env.state)

        # Normalize action
        action = action / self.action_space

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

        slope_percent = self.gradient(self.cur_position)
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

        reward = self.reward(self.state)

        if self.cur_position >= self.COURSE_DISTANCE:
            terminated = 1

        # Info
        # -------------------------------------------------

        info = self.info(
            {
                "position": self.cur_position,
                "action": action,
                "power_agent_w": power_agent_w,
                "reward": reward,
            }
        )

        # Step count
        self.step_count += 1

        return self.state, reward, terminated, truncated, info

    def reset(self, start: int = None):
        self.ghost_env.reset()
        self.ghost_action = self.ghost(self.ghost_env.state)

        self.step_count = 0
        self.cur_AWC_j = RIDER_AWC_j
        self.cur_velocity = 0
        self.cur_position = start if start else self.START_DISTANCE

        return self.state, self.info()
