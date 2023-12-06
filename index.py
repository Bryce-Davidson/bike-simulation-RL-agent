from lib.data import save_data
from gym.wrappers import FlattenObservation
import numpy as np
import keras
import time
import json

from courses import tenByOneKm, flat, rollingHills, complicated
from env import RiderEnv
from lib.JsonEncoders import NpEncoder

if __name__ == "__main__":
    env = RiderEnv(gradient=tenByOneKm, distance=4000)
    env = FlattenObservation(env)

    data = []

    observation, info = env.reset()

    while env.cur_position < env.COURSE_DISTANCE:
        observation = np.array([observation])

        action = 10

        observation, reward, terminated, truncated, info = env.step(action)

        print(json.dumps(info, indent=4, cls=NpEncoder))

        data.append(info)
        # time.sleep(1)

        save_data("test", data)
