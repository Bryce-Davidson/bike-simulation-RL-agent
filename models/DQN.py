from env import BikersEnv, NpEncoder
from courses import tenByOneKm, rollingHills, complicated
from keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout, Flatten
from gym.wrappers import FlattenObservation
import keras
import numpy as np
import random
import json

class DQNAgent:
    def __init__(self, input_dims, output_dims, batch_size=32):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size

        self.gamma = 0.1 # discount rate
        self.epsilon = 1 # initial exploration rate
        self.epsilon_min = 0.01 # minimum exploration rate
        self.epsilon_decay = 0.9995 # exploration decay rate

        self.memory_size = 1000
        self.memories = []

        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential()

        model.add(Dense(100, input_dim=self.input_dims, activation='relu'))

        model.add(Dense(100, activation='relu'))

        model.add(Dense(self.output_dims, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

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
        if len(self.memories) < self.batch_size:
            return

        # Randomly sample a batch of memories
        batch = random.sample(self.memories, self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state, terminated in batch:
            states.append(state)
            state = np.array([state])
            next_state = np.array([next_state])

            target = reward

            if not terminated:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            current = self.model.predict(state, verbose=0)
            current[0][action] = target

            targets.append(current[0])

        states = np.array(states)
        targets = np.array(targets)

        self.model.fit(states, targets, epochs=1, verbose=1)

        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            random_action = random.randrange(self.output_dims)
            print(f"Random action: {random_action}")
            return random_action

        state = np.array([state])
        act_values = self.model.predict(state, verbose=0)
        predicted_action = np.argmax(act_values[0])
        print(f"Predicted action: {predicted_action}")
        return predicted_action

    def save(self, name):
        self.model.save(name)


# ----------------------------------------------

episode_data_file_path = "episode_logs.csv"
with open(episode_data_file_path, 'w') as f:
    f.write("episode,total_reward,steps\n")


env = BikersEnv(course_fn=tenByOneKm, COURSE_DISTANCE=4000)
env = FlattenObservation(env)

# Define the input and output dimensions
input_dims = env.observation_space.shape[0]
output_dims = env.action_space.n + 1

# Create the agent
agent = DQNAgent(input_dims, output_dims, batch_size=32)

# Define the number of episodes
episodes = 600
for e in range(1, episodes+1):
    # Reset the environment and get the initial state
    cur_state, cur_info = env.reset(
        START_DISTANCE=env.COURSE_DISTANCE - e*50)

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
                "episode": e+1,
                "total_reward": total_reward,
                "steps": cur_step,
            }
            # Append the data to a file in CSV format
            with open(episode_data_file_path, 'a') as f:
                f.write(f"{data['episode']},{data['total_reward']},{data['steps']}\n")

            agent.save('DQN.keras')
            break

        cur_step += 1
