import pandas as pd
import numpy as np
import keras
import time
import json

from courses import tenByOneKm, flat, rollingHills, shortTest
from env import RiderEnv
from lib.JsonEncoders import NpEncoder

WEIGHTS_FOLDER_PATH = "./models/weights"
DATA_DIR = "./data"

# --------------------------------------------------------------


MODEL_SLUG = "TDQN-1200m-shortTest-07-12-2023_01:30"
course_name = "shortTest"
distance = 1200

model = keras.models.load_model(f"{WEIGHTS_FOLDER_PATH}/{MODEL_SLUG}.keras")

# --------------------------------------------------------------

if __name__ == "__main__":
    env = RiderEnv(gradient=shortTest, distance=distance, reward=lambda x: 0)

    data = []

    state, info = env.reset()

    while env.cur_position < env.COURSE_DISTANCE:
        state = np.array([state])

        action = np.argmax(model.predict(state, verbose=0)[0])

        state, reward, terminated, truncated, info = env.step(action)

        data.append(info)

        print(
            json.dumps(
                info,
                indent=4,
                cls=NpEncoder,
            )
        )

        time.sleep(0.01)

    # -------------------------DATA---------------------------------

    df = pd.DataFrame(data)
    df.to_csv(f"{DATA_DIR}/{course_name}-{distance}m-{MODEL_SLUG}.csv")
