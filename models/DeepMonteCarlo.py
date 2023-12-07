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
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.999999  # exploration decay rate

        # An array of state, action pairs
        self.states = []
        self.actions = []

        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(24, input_dim=self.input_dims, activation="relu"))

        model.add(Dense(24, activation="relu"))

        model.add(Dense(self.output_dims, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def forget(self):
        self.states = []
        self.actions = []

    def remember(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    def replay(self, total_reward: int):
        training_states = np.array(self.states)
        currents = self.model.predict(training_states, verbose=0)

        for i in range(len(self.states)):
            action = self.actions[i]
            currents[i][action] = total_reward

        self.model.fit(
            training_states,
            currents,
            epochs=1,
            verbose=1,
        )

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.forget()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            random_action = random.randrange(self.output_dims)

            # print(f"Random action: {random_action}")
            return random_action

        state = np.array([state])
        act_values = self.model.predict(state, verbose=0)
        predicted_action = np.argmax(act_values[0])

        # print(f"Predicted action: {predicted_action}")

        return int(predicted_action)

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
        reward = 100

    return reward


env = RiderEnv(gradient=shortTest, distance=distance, reward=reward_fn)

input_dims = len(env.observation_space)
output_dims = env.action_space.n + 1

# # Create the agent
agent = DeepMonteCarlo(input_dims, output_dims)

episodes = 100000
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
            # start to get greedy after 2000 episodes
            if e >= 2000:
                agent.epsilon_decay = 0.9995

            agent.replay(total_reward)
            agent.save(slug)

            write_row(
                log_slug,
                {
                    "episode": e,
                    "epsilon": agent.epsilon,
                    "reward": total_reward,
                    "steps": env.step_count,
                    "exit_reason": None if not terminated else "terminated",
                },
            )

            print(f"---------Episode: {e+1}-----------")
            print(f"Epsilon: {agent.epsilon}")
            print(f"Total reward: {total_reward}")
            print(f"Steps: {env.step_count}")
            print(f"Exit reason: {'terminated' if terminated else 'truncated'}")

            break
