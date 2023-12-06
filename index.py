from gym.wrappers import FlattenObservation
import pandas as pd
import numpy as np
import keras
import time
import json

from courses import tenByOneKm, flat, rollingHills, complicated
from env import BikersEnv, NpEncoder

# load the model from file

if __name__ == "__main__":
    env = BikersEnv(course_fn=tenByOneKm, COURSE_DISTANCE=4000)
    env = FlattenObservation(env)

    model = keras.models.load_model('DQN.keras')

    data = []

    observation, info = env.reset()

    while env.cur_position < env.COURSE_DISTANCE:
        observation = np.array([observation])

        action = np.argmax(model.predict(observation, verbose=0)[0])
        # if env.course_fn(env.cur_position) >= 0:
        #     action = 10
        # else:
        #     action = 8

        # action = 10

        observation, reward, terminated, truncated, info = env.step(action)

        print(json.dumps(info, indent=4, cls=NpEncoder))

        data.append(info)

        # time.sleep(1)

    df = pd.DataFrame(data)
    # df.to_csv(f'./data/rider_100_80_tenByOneKm.csv', index=False)
    # df.to_csv(f'./data/rider_100_tenByOneKm.csv', index=False)
    df.to_csv(f'./data/rider_model_tenByOneKm.csv', index=False)