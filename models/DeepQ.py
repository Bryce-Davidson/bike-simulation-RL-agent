import sys
import os

# Add the parent directory to the path so we can import the env
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.logs import write_row
from lib.JsonEncoders import NpEncoder
from env import RiderEnv
from courses import tenByOneKm, rollingHills, shortTest
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
        self.epsilon_decay = 0.99999  # exploration decay rate

        self.memory_size = 1000
        self.memories = []

        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(64, input_dim=self.input_dims, activation="relu"))

        model.add(Dense(64, activation="relu"))

        model.add(Dense(64, activation="relu"))

        model.add(Dense(self.output_dims, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def forget(self):
        self.memories = []

    def remember(self, state, action, reward, next_state, terminated):
        self.memories.append((state, action, reward, next_state, terminated))

        if len(self.memories) > self.batch_size:
            self.replay()
            self.memories.pop(0)

    def replay(self):
        if len(self.memories) < self.batch_size:
            return

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

        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            random_action = random.randrange(self.output_dims)
            # print(f"Random action: {random_action}")
            return random_action

        state = np.array([state])
        act_values = self.model.predict(state, verbose=0)
        predicted_action = np.argmax(act_values[0])
        # print(f"Predicted action: {predicted_action}")
        return predicted_action

    def save(self, name):
        self.model.save(name)


# -------------------------LOGS---------------------------------

course = "shortTest"
distance = 1200

log_path = f"../logs"

slug = f"DQN-{distance}m-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"

log_slug = f"{log_path}/{slug}.csv"

# -------------------------TRAINING-----------------------------


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

    if velocity < 0:
        reward = -100

    if percent_complete >= 1:
        reward = 1000

    return reward


env = RiderEnv(gradient=tenByOneKm, distance=distance, reward=reward_fn)

# Define the input and output dimensions
input_dims = len(env.observation_space)
output_dims = env.action_space.n + 1

# Create the agent
agent = DeepQ(input_dims, output_dims, batch_size=128)

# Define the number of episodes
episodes = 600
for e in range(1, episodes + 1):
    # Reset the environment and get the initial state
    cur_state, cur_info = env.reset()

    total_reward = 0
    while True:
        action = agent.act(cur_state)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        agent.remember(cur_state, action, reward, next_state, terminated)

        print(f"---------Episode: {e+1}, Step: {env.step_count}---------")
        print(f"Action: {action}")
        print(f"Epsilon: {agent.epsilon}")
        print(json.dumps(next_info, indent=4, cls=NpEncoder))

        cur_state, cur_info = next_state, next_info

        total_reward += reward

        if terminated or truncated:
            agent.replay(total_reward)

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

            agent.save(slug)

            print(f"---------Episode: {e+1}-----------")
            print(f"Epsilon: {agent.epsilon}")
            print(f"Total reward: {total_reward}")
            print(f"Steps: {env.step_count}")
            print(f"Exit reason: {'terminated' if terminated else 'truncated'}")

            break
