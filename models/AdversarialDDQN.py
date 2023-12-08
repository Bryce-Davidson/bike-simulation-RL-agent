import sys
import os

# Add the parent directory to the path so we can import the libraries
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import deque
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

        self.gamma = 0.9  # discount rate
        self.epsilon = 1  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.9995  # exploration decay rate

        self.memory_size = 1000
        self.memories = deque(maxlen=self.memory_size)

        self.learning_rate = 0.01
        self.model = self.build_model()
        self.target = keras.models.clone_model(self.model)
        self.replays = 0

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(100, input_dim=self.input_dims, activation="relu"))

        model.add(Dense(100, activation="relu"))

        model.add(Dense(self.output_dims, activation="linear"))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=3000,
            decay_rate=0.9,
        )

        model.compile(loss="mse", optimizer=Adam(learning_rate=lr_schedule))

        return model

    def remember(self, state, action, reward, next_state, terminated):
        self.memories.append((state, action, reward, next_state, terminated))

        if len(self.memories) > self.batch_size:
            self.replay()

    def replay(self):
        batch = random.sample(self.memories, self.batch_size)

        states, actions, rewards, next_states, terminals = zip(*batch)

        states = np.array(states)

        # Predict states using the current network
        currents = self.model.predict(states, verbose=0)
        # Predict next states using the target network
        nexts = self.target.predict(np.array(next_states), verbose=0)

        # Update the current network with the new values
        for i, action in enumerate(actions):
            currents[i][action] = rewards[i]

            if not terminals[i]:
                currents[i][action] += self.gamma * np.amax(nexts[i])

        self.model.fit(states, currents, epochs=1, verbose=1)

        # Update the target network every 100 steps
        if self.replays % 100 == 0:
            self.target.set_weights(self.model.get_weights())

        # Update the epsilon value
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

slug = f"DDQN-{distance}m-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"

log_slug = f"{log_path}/{slug}.csv"


# -------------------------REWARDS--------------------------

ghost = RiderEnv(gradient=testCourse, distance=distance, reward=lambda x: 0)


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

    if agent_percent_complete >= 1 and ghost_percent_complete < 1:
        return 10000000

    reward = -1

    diff = agent_percent_complete - ghost_percent_complete

    reward += diff if diff > 0 else diff * 1000

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
agent = DeepQ(input_dims, output_dims, batch_size=64)

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

            break
