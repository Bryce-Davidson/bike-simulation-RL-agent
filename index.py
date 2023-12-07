import pandas as pd
import numpy as np
import keras
import time
import json

from courses import tenByOneKm, flat, rollingHills, complicated
from env import RiderEnv
from lib.JsonEncoders import NpEncoder

WEIGHTS_FOLDER_PATH = "./models/weights"
MODEL_SLUG = "DMC-100m-tenByOneKm-06-12-2023_13:49"

# load the model from the .keras file
model = keras.models.load_model(f"{WEIGHTS_FOLDER_PATH}/{MODEL_SLUG}.keras")

FORCED_ACTION = 10

if __name__ == "__main__":
    env = RiderEnv(gradient=tenByOneKm, distance=10_000)

    data = []

    state, info = env.reset()

    while env.cur_position < env.COURSE_DISTANCE:
        state = np.array([state])

        if FORCED_ACTION:
            action = FORCED_ACTION
        else:
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

        time.sleep(0.1)

    # Save the data to a csv file

    df = pd.DataFrame(data)
    if FORCED_ACTION:
        df.to_csv(f"./models/forced_action_{FORCED_ACTION}.csv")
    else:
        df.to_csv(f"./models/{MODEL_SLUG}.csv")
