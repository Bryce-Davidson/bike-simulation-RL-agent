import pandas as pd
import numpy as np
import keras
import time
import json

from courses import tenByOneKm, flat, rollingHills, shortTest
from env import RiderEnv
from lib.JsonEncoders import NpEncoder

DATA_DIR = "./data"

# --------------------------------------------------------------

course_name = "shortTest"
distance = 1200

# --------------------------------------------------------------

if __name__ == "__main__":
    env = RiderEnv(gradient=shortTest, distance=distance, reward=lambda x: 0)

    data = []

    state, info = env.reset()

    while env.cur_position < env.COURSE_DISTANCE:
        state = np.array([state])

        if env.gradient(env.cur_position) > 0:
            action = 10
        else:
            action = 8

        state, reward, terminated, truncated, info = env.step(action)

        data.append(info)

        print(
            json.dumps(
                info,
                indent=4,
                cls=NpEncoder,
            )
        )

        # time.sleep(0.1)

    # -------------------------DATA---------------------------------

    df = pd.DataFrame(data)
    df.to_csv(f"{DATA_DIR}/{course_name}-{distance}-forced_action_10:8.csv")
