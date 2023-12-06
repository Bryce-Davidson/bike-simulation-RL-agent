import datetime
from lib.logs import write_episode_data_to_csv
from env import RiderEnv, NpEncoder
from courses import tenByOneKm, rollingHills, complicated
from keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout
import keras
import numpy as np
import random
import json


class DeepMonteCarlo:
    def __init__(self, input_dims: int, output_dims: int):
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.gamma = 0.9  # discount rate
        self.epsilon = 1  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.9995  # exploration decay rate

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

    def remember(self, state, action, reward, next_state, terminated):
        self.memories.append((state, action, reward, next_state, terminated))

    def replay(self, total_reward: int):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            random_action = random.randrange(self.output_dims)

            return random_action

        state = np.array([state])
        act_values = self.model.predict(state, verbose=0)
        predicted_action = np.argmax(act_values[0])

        return predicted_action

    def save(self, name):
        self.model.save(name)


# -------------------------LOGGING-----------------------------

episode_data_file_path = f"./logs/DeepMomteCarlo/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_episodes.csv"

# -------------------------TRAINING-----------------------------

env = RiderEnv(gradient=rollingHills, distance=3000)

input_dims = env.observation_space.shape[0]
output_dims = env.action_space.n + 1

# Create the agent
agent = DeepMonteCarlo(input_dims, output_dims)

episodes = 600
for e in range(episodes):
    cur_state, cur_info = env.reset()

    total_reward = 0
    while True:
        print(f"---------Episode: {e+1}, Step: {env.step_count}---------")
        print(f"Epsilon: {agent.epsilon}")
        print(json.dumps(cur_info, indent=4, cls=NpEncoder))

        action = agent.act(cur_state)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        agent.remember(cur_state, action, reward, next_state, terminated)

        cur_state, cur_info = next_state, next_info

        total_reward += reward

        if terminated or truncated:
            agent.replay(total_reward)

            write_episode_data_to_csv(
                episode_data_file_path,
                total_reward,
                episode_number=e,
                steps=env.step_count,
                exit_reason="terminated" if terminated else "truncated",
            )
            break
