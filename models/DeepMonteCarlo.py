import sys
import os

# Add the parent directory to the path so we can import the env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import datetime
from lib.logs import write_row
from env import RiderEnv
from courses import shortTest
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

        self.epsilon = 1  # initial exploration rate
        self.epsilon_min = 0.001  # minimum exploration rate
        self.epsilon_decay = 1  # exploration decay rate

        # An array of state, action pairs
        self.memory_limit = 1000
        self.memories = []

        self.learning_rate = 0.01
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(500, input_dim=self.input_dims, activation="relu"))

        model.add(Dense(500, activation="relu"))

        model.add(Dense(500, activation="relu"))

        model.add(Dense(self.output_dims, activation="linear"))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate, decay_steps=100, decay_rate=0.96
        )

        model.compile(loss="mse", optimizer=Adam(learning_rate=lr_schedule))

        return model

    def forget(self):
        self.memories = []

    def remember(self, state, action):
        self.memories.append((state, action))

    def replay(self, total_reward: int):
        states, actions = zip(*self.memories)

        states = np.array(states)
        currents = self.model.predict(states, verbose=0)

        for i, action in enumerate(actions):
            currents[i][action] = total_reward

        # Update the model
        self.model.fit(
            states,
            currents,
            epochs=1,
            verbose=1,
        )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.forget()

        # 3 return the current learning rate of the network
        return self.model.optimizer.lr

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.output_dims)

        state = np.expand_dims(state, axis=0)
        return int(np.argmax(self.model.predict(state, verbose=0)[0]))

    def save(self, slug):
        self.model.save(filepath=f"./weights/{slug}.keras")


# -------------------------LOGS---------------------------------

course = "shortTest"
distance = 1200

log_path = f"../logs"

slug = f"DMC-{distance}m-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"

log_slug = f"{log_path}/{slug}.csv"

# -------------------------TRAINING-----------------------------


def reward_fn(state):
    power_max_w, velocity, gradient, percent_complete, AWC = state

    reward = -1

    if percent_complete >= 0.95:
        reward += -AWC * 0.01

    return reward


env = RiderEnv(gradient=shortTest, distance=distance, reward=reward_fn)

input_dims = len(env.observation_space)
output_dims = env.action_space.n + 1

# # Create the agent
agent = DeepMonteCarlo(input_dims, output_dims)

episodes = 100000
for e in range(episodes):
    if e >= 15000:
        agent.epsilon = 0

    cur_state, cur_info = env.reset()

    total_reward = 0
    while True:
        action = agent.act(cur_state)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        agent.remember(cur_state, action)

        cur_state, cur_info = next_state, next_info

        total_reward += reward

        if terminated or truncated:
            lr = agent.replay(total_reward)
            agent.save(slug)

            write_row(
                log_slug,
                {
                    "episode": e,
                    "learning_rate": lr,
                    "epsilon": agent.epsilon,
                    "reward": total_reward,
                    "steps": env.step_count,
                    "exit_reason": "terminated" if terminated else "truncated",
                },
            )

            print(f"---------Episode: {e}-----------")
            print(f"Total reward: {total_reward}")
            print(f"Learning rate: {lr}")
            print(f"Epsilon: {agent.epsilon}")
            print(f"Steps: {env.step_count}")
            print(f"Exit reason: {'terminated' if terminated else 'truncated'}")

            break
