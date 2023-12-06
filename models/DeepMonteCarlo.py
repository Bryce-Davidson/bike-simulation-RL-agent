from env import RiderEnv, NpEncoder
from courses import tenByOneKm, rollingHills, complicated
from keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout
import keras
import numpy as np
import random
import json


class DeepMonteCarlo:
    def __init__(self, input_dims: int, output_dims: int, batch_size: int = 32):
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

        model.add(Dense(self.output_dims, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def forget(self):
        self.memories = []

    def remember(self, state, action, reward, next_state, terminated):
        self.memories.append((state, action, reward, next_state, terminated))

        if len(self.memories) > self.memory_size:
            self.memories.pop(0)

        if len(self.memories) > self.batch_size:
            self.replay()

    def replay(self):
        # Error handling
        # ----------------------------------------------
        if len(self.memories) < self.batch_size:
            raise ValueError(
                f"Batch size ({self.batch_size}) is larger than memory size ({len(self.memories)})"
            )

        # ----------------------------------------------

        # Decay the exploration rate
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


# ----------------------------------------------

episode_data_file_path = "episode_logs.csv"
with open(episode_data_file_path, "w") as f:
    f.write("episode,total_reward,steps\n")


env = RiderEnv(gradient=tenByOneKm, distance=4000)

# Define the input and output dimensions
input_dims = env.observation_space.shape[0]
output_dims = env.action_space.n + 1

# Create the agent
agent = DeepMonteCarlo(input_dims, output_dims, batch_size=32)

# Define the number of episodes
episodes = 600
for e in range(1, episodes + 1):
    # Reset the environment and get the initial state
    cur_state, cur_info = env.reset(START_DISTANCE=env.COURSE_DISTANCE - e * 50)

    total_reward = 0
    cur_step = 0
    while True:
        action = agent.act(cur_state)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        agent.remember(cur_state, action, reward, next_state, terminated)

        print(f"---------Episode: {e+1}, Step: {cur_step}---------")
        print(f"Action: {action}")
        print(f"Epsilon: {agent.epsilon}")
        print(json.dumps(next_info, indent=4, cls=NpEncoder))

        cur_state, cur_info = next_state, next_info

        total_reward += reward

        if terminated:
            data = {
                "episode": e + 1,
                "total_reward": total_reward,
                "steps": cur_step,
            }
            # Append the data to a file in CSV format
            with open(episode_data_file_path, "a") as f:
                f.write(f"{data['episode']},{data['total_reward']},{data['steps']}\n")

            agent.save("DQN.keras")
            break

        cur_step += 1
