import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
from lib.logs import write_episode_data_to_csv
from env import RiderEnv
from courses import tenByOneKm, rollingHills, complicated
from keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout
import keras
import numpy as np
import random

# stop numpy from printing in scientific notation
np.set_printoptions(suppress=True)


class DeepMonteCarlo:
    def __init__(self, input_dims: int, output_dims: int):
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.gamma = 0.1  # discount rate
        self.epsilon = 1  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.9  # exploration decay rate

        # An array of state, action pairs
        self.memories = []

        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(100, input_dim=self.input_dims, activation="relu"))

        model.add(Dense(100, activation="relu"))

        model.add(Dense(self.output_dims, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def forget(self):
        self.memories = []

    def remember(self, state, action):
        self.memories.append((state, action))

    def replay(self, total_reward: int):
        states = []
        targets = []

        for i, (state, action) in enumerate(self.memories):
            states.append(state)
            current = self.model.predict(np.array([state]), verbose=0)
            current[0][action] = total_reward
            targets.append(current)

        self.model.fit(
            np.array(states),
            np.array(targets),
            epochs=1,
            verbose=1,
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.forget()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            random_action = random.randrange(self.output_dims)

            return random_action

        state = np.array([state])
        act_values = self.model.predict(state, verbose=0)
        predicted_action = np.argmax(act_values[0])

        return int(predicted_action)

    def save(self, name):
        self.model.save(filepath=f"./weights/{name}.keras")


# -------------------------LOGGING-----------------------------

course = "tenByOneKm"
distance = 100
log_slug = (
    f"DMC-{distance}km-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"
)

# -------------------------TRAINING-----------------------------

env = RiderEnv(gradient=tenByOneKm, distance=distance)

input_dims = len(env.observation_space)
output_dims = env.action_space.n + 1

# # Create the agent
agent = DeepMonteCarlo(input_dims, output_dims)

episodes = 10000
for e in range(episodes):
    cur_state, cur_info = env.reset()

    total_reward = 0
    while True:
        action = agent.act(cur_state)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        agent.remember(cur_state, action)

        cur_state, cur_info = next_state, next_info

        total_reward += reward

        if terminated or truncated:
            agent.replay(total_reward)

            write_episode_data_to_csv(
                log_slug,
                total_reward,
                episode_number=e,
                steps=env.step_count,
                exit_reason="terminated" if terminated else "truncated",
            )

            agent.save(log_slug)

            print(f"---------Episode: {e+1}-----------")
            print(f"Epsilon: {agent.epsilon}")
            print(f"Total reward: {total_reward}")
            print(f"Steps: {env.step_count}")
            print(f"Exit reason: {'terminated' if terminated else 'truncated'}")

            break
