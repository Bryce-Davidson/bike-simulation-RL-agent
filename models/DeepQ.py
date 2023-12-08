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

np.set_printoptions(suppress=True)


class DeepQ:
    def __init__(self, input_dims, output_dims, batch_size=32):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size

        self.gamma = 0.1  # discount rate
        self.epsilon = 1  # initial exploration rate
        self.epsilon_min = 0.001  # minimum exploration rate
        self.epsilon_decay = 0.999999  # exploration decay rate

        self.memory_size = 1000
        self.memories = []

        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(500, input_dim=self.input_dims, activation="relu"))

        model.add(Dense(500, activation="relu"))

        model.add(Dense(500, activation="relu"))

        model.add(Dense(500, activation="relu"))

        model.add(Dense(self.output_dims, activation="linear"))

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

        states, actions, rewards, next_states, terminals = zip(*batch)

        states = np.array(states)

        currents = self.model.predict(states, verbose=0)
        nexts = self.model.predict(np.array(next_states), verbose=0)

        for i, action in enumerate(actions):
            currents[i][action] = rewards[i]

            if not terminals[i]:
                currents[i][action] += self.gamma * np.amax(nexts[i])

        self.model.fit(states, currents, epochs=1, verbose=1)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dims)

        state = np.array([state])
        return np.argmax(self.model.predict(state, verbose=0)[0])

    def save(self, slug):
        self.model.save(filepath=f"./weights/{slug}.keras")


# -------------------------LOGS---------------------------------

course = "testCourse"
distance = 400

log_path = f"../logs"

slug = f"DQN-{distance}m-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"

log_slug = f"{log_path}/{slug}.csv"


# -------------------------REWARDS--------------------------

ghost = RiderEnv(gradient=testCourse, distance=distance, reward=lambda x: 0)
ghost.reset()


def reward_fn(state):
    (
        agent_power_max_w,
        agent_velocity,
        agent_gradient,
        agent_percent_complete,
        AWC,
    ) = state

    (
        ghost_power_max_w,
        ghost_velocity,
        ghost_gradient,
        ghost_percent_complete,
        ghost_AWC,
    ) = ghost.step(10)[0]

    if ghost_percent_complete >= 1 and agent_percent_complete <= 1:
        return -100

    if agent_percent_complete >= 1 and ghost_percent_complete < 1:
        return 1000

    reward = (agent_percent_complete - ghost_percent_complete) * 100

    if agent_velocity < 0:
        reward -= 100

    if agent_percent_complete >= 1:
        reward += 100

    return reward


# -------------------------TRAINING-----------------------------

env = RiderEnv(gradient=testCourse, distance=distance, reward=reward_fn)

# Define the input and output dimensions
input_dims = len(env.observation_space)
output_dims = env.action_space.n + 1

# Create the agent
agent = DeepQ(input_dims, output_dims, batch_size=128)

# Define the number of episodes
episodes = 100000
for e in range(0, episodes):
    ghost.reset()
    cur_state, cur_info = env.reset()

    total_reward = 0
    while True:
        print(f"---------Episode: {e+1}/{episodes}, Step: {env.step_count}-----------")
        action = agent.act(cur_state)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        total_reward += reward

        agent.remember(cur_state, action, reward, next_state, terminated)

        print(
            json.dumps(
                {
                    **next_info,
                    "total_reward": total_reward,
                    "epsilon": agent.epsilon,
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
