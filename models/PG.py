import sys
import os

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


class PolicyGradientAgent:
    def __init__(self, input_dims, output_dims, learning_rate=0.01):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.states = []
        self.actions = []
        self.rewards = []

    def build_model(self):
        model = keras.models.Sequential()
        model.add(Dense(24, input_dim=self.input_dims, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.output_dims, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def act(self, state):
        state = state.expand_dims(axis=0)
        action = self.model.predict(state, batch_size=1)[0]
        return action

    def train(self):
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)

        # Normalize rewards
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        # Train the model
        self.model.fit(states, actions, sample_weight=rewards, epochs=1, verbose=0)

        # Reset states, actions and rewards
        self.states = []
        self.actions = []
        self.rewards = []


# -------------------------LOGS---------------------------------

course = "shortTest"
distance = 1200

log_path = f"./logs"

slug = f"DQN-{distance}m-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H_%M')}"

log_slug = f"{log_path}/{slug}.csv"

# -------------------------TRAINING-----------------------------


def potential(state):
    power_max_w, velocity, gradient, percent_complete, AWC = state
    # Encourage the agent to complete the course quickly and maintain AWC
    return percent_complete + np.log(AWC + 1)


def reward_fn(state, next_state):
    power_max_w, velocity, gradient, percent_complete, AWC = state

    reward = -1

    if velocity < 0:
        reward += -100

    if percent_complete >= 1:
        reward += 1000

    reward += potential(next_state) - potential(state)

    return reward


agent = PolicyGradientAgent(input_dims=4, output_dims=2, learning_rate=0.001)

env = RiderEnv(gradient=tenByOneKm, distance=distance, reward=reward_fn)

for i_episode in range(500):
    state = env.reset()
    for t in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward)
        state = next_state
        if done:
            agent.train()
            break
