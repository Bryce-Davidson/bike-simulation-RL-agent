from gym.wrappers import FlattenObservation
import gym
import pandas as pd
import numpy as np
from gym.spaces import Dict, Box, Discrete
import time
import json

RIDER_AWC_MIN = 0 # in watts/kg
RIDER_AWC_MAX = 10000 # in watts/kg

RIDER_VELOCITY_MIN = -28 # in m/s
RIDER_VELOCITY_MAX = 28 # in m/s

NUM_ACTIONS = 10

MAX_AWC = 9758 # in joules
CP_w = 234 # in watts
RIDER_MASS_kg = 70 # in kilos

G = 9.81 # in m/s^2
ROLLING_RESISTANCE_COEFF = 0.0046

def riderPowerMax(AWC_j):
    return 7e-6*AWC_j**2 + 0.0023*AWC_j + CP_w

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class BikersEnv(gym.Env):
    def __init__(self, course_fn, COURSE_DISTANCE=5000, START_DISTANCE=0):
        self.render_mode = None
        self.course_fn = course_fn

        self.START_DISTANCE = START_DISTANCE
        self.COURSE_DISTANCE = COURSE_DISTANCE

        self.observation_space = Dict({
                "power_max_w": Box(low=0, high=3000, shape=(1,), dtype=float),
                "velocity": Box(low=RIDER_VELOCITY_MIN, high=RIDER_VELOCITY_MAX, shape=(1,), dtype=float),
                "gradient": Box(low=-100, high=100, shape=(1,), dtype=float),
                "percent_complete" : Box(low=0, high=1, shape=(1,), dtype=float),
                "AWC": Box(low=RIDER_AWC_MIN, high=RIDER_AWC_MAX, shape=(1,), dtype=float),
            })

        self.action_space = Discrete(NUM_ACTIONS)
        self.time_step = 0

        # Agent variables
        # ---------------------------
        self.cur_AWC_j = MAX_AWC
        self.cur_velocity = 0
        self.cur_position = self.START_DISTANCE

    def step(self, action: int):
        # Normalize the action
        action = action/NUM_ACTIONS

        # Physiological calculations
        # -------------------------------------------------

        power_max_w = riderPowerMax(self.cur_AWC_j)

        power_agent_w = action*power_max_w

        if power_agent_w > CP_w:
            # Fatigue
            fatigue = -(power_agent_w - CP_w)
        else:
            # Recovery
            p_agent_adj_w = 0.0879*power_agent_w + 204.5
            fatigue = -(p_agent_adj_w - CP_w)

        self.cur_AWC_j = min(self.cur_AWC_j + fatigue, MAX_AWC)

        # Environment/Agent calculations
        # -------------------------------------------------

        # Resistance calculations
        # -----------------------

        slope_percent = self.course_fn(self.cur_position)
        slope_radians = np.arctan(slope_percent/100)

        horizontal_gravity_force = G*RIDER_MASS_kg*np.sin(slope_radians)

        rolling_resistance_force = G*RIDER_MASS_kg*ROLLING_RESISTANCE_COEFF

        drag_force = 0.19*(self.cur_velocity)**2

        total_resistance_w = (horizontal_gravity_force + rolling_resistance_force + drag_force)*self.cur_velocity

        # Velocity calculations
        # ---------------------

        cur_KE_j = (1/2)*RIDER_MASS_kg*(self.cur_velocity)**2

        final_KE_j = cur_KE_j - total_resistance_w + power_agent_w

        self.cur_velocity = np.sign(final_KE_j) * np.sqrt(2*abs(final_KE_j)/RIDER_MASS_kg)


        # Rider position calculations
        # ---------------------------

        new_position = self.cur_position + self.cur_velocity

        self.cur_position = min(new_position, self.COURSE_DISTANCE)

        # New Observation

        observation = {
                    "power_max_w": power_max_w,
                    "velocity" : self.cur_velocity,
                    "gradient": slope_percent,
                    "percent_complete" : self.cur_position/self.COURSE_DISTANCE,
                    "AWC": self.cur_AWC_j,
                }

        # Rewards and termination
        # -----------------------

        terminated = 0

        reward = -1

        if self.cur_position >= self.COURSE_DISTANCE:
            reward += 100
            terminated = 1

        if self.cur_velocity <= 0:
            reward -= 100

        # Info
        # -----------------------

        info = {
            **observation,
            "position": self.cur_position,
            "action": action,
            "power_output" : power_agent_w,
            "reward": reward,
        }

        self.time_step += 1

        truncated = 0
        return observation, reward, terminated, truncated, info

    def render(self):
        return

    def reset(self, START_DISTANCE=None):
        self.cur_AWC_j = MAX_AWC
        self.cur_velocity = 0

        self.cur_position = START_DISTANCE if START_DISTANCE else self.START_DISTANCE

        observation = {
            "power_max_w": riderPowerMax(self.cur_AWC_j),
            "velocity" : self.cur_velocity,
            "gradient": self.course_fn(self.cur_position),
            "percent_complete" : self.cur_position/self.COURSE_DISTANCE,
            "AWC": self.cur_AWC_j,
            }

        return observation, None