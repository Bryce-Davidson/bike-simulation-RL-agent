from lib.data import save_data
from gym.wrappers import FlattenObservation
import numpy as np
import keras
import time
import json

from courses import tenByOneKm, flat, rollingHills, complicated
from env import RiderEnv
from lib.JsonEncoders import NpEncoder

WEIGHTS_FOLDER_PATH = "./models/weights/"
MODEL_WEIGHTS_FILE_NAME = "DMC-100m-tenByOneKm-06-12-2023_13:49.keras"

# load the model from the .keras file
model = keras.models.load_model(f"{WEIGHTS_FOLDER_PATH}/{MODEL_WEIGHTS_FILE_NAME}")

if __name__ == "__main__":
    env = RiderEnv(gradient=tenByOneKm, distance=100)

    data = []

    state, info = env.reset()

    while env.cur_position < env.COURSE_DISTANCE:
        state = np.array([state])

        action_values = model.predict(state, verbose=0)[0]
        action = np.argmax(action_values)

        # action = 10

        state, reward, terminated, truncated, info = env.step(action)

        print(
            json.dumps(
                {
                    **info,
                    # "all_action_values": action_values,
                    "predicted_reward": action_values[action],
                },
                indent=4,
                cls=NpEncoder,
            )
        )

        data.append(info)
        time.sleep(0.11)

        save_data("100_rider", data)
