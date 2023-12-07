import sys
import os

# Add the parent directory to the path so we can import the libraries
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.logs import write_row
from lib.JsonEncoders import NpEncoder
from env import RiderEnv
from courses import testCourse
from keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout, Flatten
from gym.wrappers import FlattenObservation
import datetime
import keras
import numpy as np
import random
import json


class DeepQ:
    def __init__(self, input_dims, output_dims, batch_size=32):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size

        self.gamma = 0.9  # discount rate
        self.epsilon = 1  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.9995  # exploration decay rate

        self.memory_size = 1000
        self.memories = []

        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(100, input_dim=self.input_dims, activation="relu"))

        model.add(Dense(100, activation="relu"))

        model.add(Dense(100, activation="relu"))

        model.add(Dense(100, activation="relu"))

        model.add(Dense(100, activation="relu"))

        model.add(Dense(50, activation="relu"))

        model.add(Dense(50, activation="relu"))

        model.add(Dense(self.output_dims, activation="linear"))

        # add in a learning rate scheduler

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            decay_rate=0.9,
        )

        model.compile(loss="mse", optimizer=Adam(learning_rate=lr_schedule))

        return model

    def forget(self):
        self.memories = []

    def remember(self, state, action, reward, next_state, terminated):
        self.memories.append((state, action, reward, next_state, terminated))

        if len(self.memories) > self.batch_size:
            self.replay()
            self.memories.pop(0)

    def replay(self):
        # Randomly sample a batch of memories
        batch = random.sample(self.memories, self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, terminated in batch:
            states.append(state)

            target = reward
            if not terminated:
                target = reward + self.gamma * np.amax(
                    self.model.predict(np.array([next_state]), verbose=0)[0]
                )

            current = self.model.predict(np.array([state]), verbose=0)
            current[0][action] = target

            targets.append(current[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=1)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dims)

        state = np.array([state])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def save(self, slug):
        self.model.save(filepath=f"./weights/{slug}.keras")


def reward_fn(state):
    """
    state = [
        power_max_w,
        velocity,
        gradient,
        percent_complete,
        AWC
    ]
    """

    power_max_w, velocity, gradient, percent_complete, AWC = state

    reward = -1

    if percent_complete >= 1:
        reward += 1000000

    return reward


# -------------------------LOGS---------------------------------

course = "testCourse"
distance = 400

log_path = f"../logs"

slug = f"DQN-{distance}m-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"

log_slug = f"{log_path}/{slug}.csv"

# -------------------------TRAINING-----------------------------


env = RiderEnv(gradient=testCourse, distance=distance, reward=reward_fn)

# Define the input and output dimensions
input_dims = len(env.observation_space)
output_dims = env.action_space.n + 1

# Create the agent
agent = DeepQ(input_dims, output_dims, batch_size=64)

# Define the number of episodes
episodes = 10000
for e in range(0, episodes):
    cur_state, cur_info = env.reset()

    total_reward = 0
    while True:
        print(f"---------Episode: {e+1}, Step: {env.step_count}-----------")
        print(f"Epsilon: {agent.epsilon}")
        action = agent.act(cur_state)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        total_reward += reward

        agent.remember(cur_state, action, reward, next_state, terminated)

        print(
            json.dumps(
                {
                    **next_info,
                    "total_reward": total_reward,
                },
                indent=4,
                cls=NpEncoder,
            )
        )

        cur_state, cur_info = next_state, next_info

        if terminated or truncated:
            agent.save(slug)
            write_row(
                log_slug,
                {
                    "episode": e,
                    "epsilon": agent.epsilon,
                    "reward": total_reward,
                    "steps": env.step_count,
                    "exit_reason": "terminated" if terminated else "truncated",
                },
            )

            print(f"---------Episode: {e+1}-----------")
            print(f"Epsilon: {agent.epsilon}")
            print(f"Total reward: {total_reward}")
            print(f"Steps: {env.step_count}")
            print(f"Exit reason: {'terminated' if terminated else 'truncated'}")

            break
