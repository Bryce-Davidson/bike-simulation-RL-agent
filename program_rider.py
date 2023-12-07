import pandas as pd
import numpy as np
import keras
import time
import json

from courses import testCourse
from env import RiderEnv
from lib.JsonEncoders import NpEncoder

DATA_DIR = "./data"


def rider100(grad):
    return 10


def rider100_80(grad):
    action = 10
    if grad < 0:
        action = 8
    return action


# --------------------------------------------------------------

course_name = "testCourse"
distance = 400

# --------------------------------------------------------------

if __name__ == "__main__":
    models = [(rider100, "10"), (rider100_80, "10:8")]

    for model, model_name in models:
        env = RiderEnv(gradient=testCourse, distance=distance, reward=lambda x: 0)

        data = []

        state, info = env.reset()

        while env.cur_position < env.COURSE_DISTANCE:
            state = np.array([state])

            action = model(env.gradient(env.cur_position))

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
        df.to_csv(f"{DATA_DIR}/{course_name}-{distance}m-{model_name}-rider.csv")
