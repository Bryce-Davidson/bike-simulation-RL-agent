import pandas as pd
import numpy as np
import keras
import time
import json

from courses import testCourse
from env import RiderEnv
from lib.JsonEncoders import NpEncoder

WEIGHTS_FOLDER_PATH = "./models/trained"
DATA_DIR = "./data"

# --------------------------------------------------------------

MODEL_SLUG = "DDQN-400m-testCourse-09-12-2023_17:47"
course_name = "testCourse"
distance = 400

model = keras.models.load_model(f"{WEIGHTS_FOLDER_PATH}/{MODEL_SLUG}/weights.keras")

# --------------------------------------------------------------

if __name__ == "__main__":
    env = RiderEnv(gradient=testCourse, distance=distance, reward=lambda x: 0)

    data = []

    state, info = env.reset()
    while True:
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

        if terminated or truncated:
            df = pd.DataFrame(data)
            df.to_csv(f"{DATA_DIR}/{course_name}-{distance}m-{MODEL_SLUG}.csv")
            break
