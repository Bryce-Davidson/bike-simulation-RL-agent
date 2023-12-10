import sys
import os

# Add the parent directory to the path so we can import the libraries
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import deque
from lib.logs import write_row
from lib.JsonEncoders import NpEncoder
from ghostEnv import GhostEnv
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

# Set the random seed for reproducibility
random.seed(0)


class DDQN:
    def __init__(
        self,
        input_dims,
        output_dims,
        dense_layers=[200, 200, 200],
        dropout=0.2,
        gamma=0.9,
        copy_ghost_replays=0,
        epsilon_start=0.5,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        target_replays=100,
        memory_size=10000,
        batch_size=64,
        lr_start=0.01,
        lr_decay=0.9,
        lr_decay_steps=3000,
    ):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.dense_layers = dense_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.target_replays = target_replays

        self.copy_ghost_replays = copy_ghost_replays
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_min = epsilon_min  # minimum exploration probability
        self.epsilon_decay = epsilon_decay  # exploration decay rate

        self.memory_size = memory_size
        self.memories = deque(maxlen=self.memory_size)
        self.replays = 0

        self.lr_start = lr_start
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.model = self.build_model()
        self.target = keras.models.clone_model(self.model)

    def build_model(self):
        model = keras.models.Sequential()

        for i, layer in enumerate(self.dense_layers):
            if i == 0:
                model.add(Dense(layer, input_dim=self.input_dims, activation="relu"))
            else:
                model.add(Dense(layer, activation="relu"))

            if self.dropout > 0:
                model.add(Dropout(self.dropout))

        model.add(Dense(self.output_dims, activation="linear"))

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr_start,
            decay_steps=self.lr_decay_steps,
            decay_rate=self.lr_decay,
            staircase=True,
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
        if self.replays % self.target_replays == 0:
            self.target.set_weights(self.model.get_weights())

        # Update the epsilon value
        if self.replays >= self.copy_ghost_replays:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Increment the number of replays
        self.replays += 1

    def act(self, state, ghost_action):
        if self.replays < self.copy_ghost_replays:
            return ghost_action

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_dims)

        state = np.array([state])
        return np.argmax(self.model.predict(state, verbose=0)[0])


# -------------------------LOGS---------------------------------

course = "testCourse"
distance = 400

TRAINED_PATH = f"./trained"

MODEL_SLUG = (
    f"ADDQN-{distance}m-{course}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}"
)

# -------------------------REWARDS--------------------------


def reward_fn(state):
    (
        agent_power_max_w,
        agent_velocity,
        agent_gradient,
        agent_percent_complete,
        AWC,
        ghost_power_max_w,
        ghost_velocity,
        ghost_gradient,
        ghost_percent_complete,
        ghost_AWC,
    ) = state

    reward = -1

    diff = agent_percent_complete - ghost_percent_complete
    reward += diff * 100 if diff > 0 else diff

    if agent_percent_complete >= 1 and ghost_percent_complete < 1:
        reward += 1000

    if agent_velocity < 0:
        reward -= 100

    return reward


# -------------------------TRAINING-----------------------------


def ghostRider(state) -> int:
    (
        ghost_power_max_w,
        ghost_velocity,
        ghost_gradient,
        ghost_percent_complete,
        ghost_AWC,
    ) = state

    if ghost_gradient >= 0:
        return 20
    else:
        return int(20 * 0.8)


env = GhostEnv(
    ghost=ghostRider,
    gradient=testCourse,
    distance=distance,
    reward=reward_fn,
    num_actions=20,
)

# Create the agent
agent = DDQN(
    input_dims=env.state.size,
    output_dims=env.action_space + 1,
    dense_layers=[12, 12],
    dropout=0.1,
    gamma=0.9,
    copy_ghost_replays=0,
    epsilon_start=1,
    epsilon_decay=0.9999,
    epsilon_min=0.01,
    target_replays=50,
    memory_size=20000,
    batch_size=64,
    lr_start=0.001,
    lr_decay=0.9,
    lr_decay_steps=5000,
)

# Define the number of episodes
episodes = 100000
for e in range(0, episodes):
    cur_state, cur_info = env.reset()

    total_reward = 0
    while True:
        print(f"---------Episode: {e}/{episodes}, Step: {env.step_count}-----------")

        ghost_action = env.ghost_action
        action = agent.act(cur_state, env.ghost_action)

        next_state, reward, terminated, truncated, next_info = env.step(action)

        write_row(
            path=f"{TRAINED_PATH}/{MODEL_SLUG}/logs/actions.csv",
            data={
                **next_info,
                "episode": e,
                "action": action,
                "ghost_action": ghost_action,
                "step": env.step_count,
                "total_reward": total_reward,
            },
        )

        total_reward += reward

        agent.remember(cur_state, action, reward, next_state, terminated or truncated)

        print(
            json.dumps(
                {
                    **next_info,
                    "ghost_action": env.ghost_action,
                    "total_reward": total_reward,
                    "replays": agent.replays,
                    "epsilon": agent.epsilon,
                },
                indent=4,
                cls=NpEncoder,
            )
        )

        cur_state, cur_info = next_state, next_info

        if terminated or truncated:
            if not os.path.exists(TRAINED_PATH):
                os.makedirs(TRAINED_PATH)

            agent.model.save(f"{TRAINED_PATH}/{MODEL_SLUG}/weights.keras")

            write_row(
                path=f"{TRAINED_PATH}/{MODEL_SLUG}/logs/episodes.csv",
                data={
                    "episode": e,
                    "epsilon": agent.epsilon,
                    "total_reward": total_reward,
                    "steps": env.step_count,
                    "exit_reason": "terminated" if terminated else "truncated",
                },
            )

            break
