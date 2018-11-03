import tensorflow as tf
# from tensorflow import keras
import numpy as np
from viz import *
from reward import *
import random
from collections import deque
import os
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from gridworld import *
from value_iteration import *


class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def load(self, name):
        self.model.load_weights(name)

    def get_max_value(self, state):
        action_values = self.model.predict(state)
        max_value = np.max(action_values[0])
        return max_value

    def __call__(self, state):
        action_values = self.model.predict(state)
        action_index = np.argmax(action_values[0])
        return action_index


if __name__ == '__main__':
    env = GridWorld("test", nx=20, ny=20)
    sheep_states = [(5, 5)]
    obstacle_states = []
    env.add_obstacles(obstacle_states)
    env.add_terminals(sheep_states)
    obstacles = {s: -10 for s in obstacle_states}
    sheep = {s: 100 for s in sheep_states}
    env.add_feature_map("sheep", sheep, default=0)

    S = tuple(it.product(range(env.nx), range(env.ny)))
    A = ((1, 0), (0, 1), (-1, 0), (0, -1))
    action_size = len(A)

    num_opisodes = 100
    for e in range(num_opisodes):
        wolf_state = random.choice(S)
        done = wolf_state in env.terminals
        while not done:
            state_img = state_to_image_array(env,
                                             [wolf_state], sheep_states, obstacle_states)
            state_size = state_img.flatten().shape[0]

            agent = DQN(state_size, action_size)

            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "linux/v100/save")
            name = str(sheep_states) + '_episode_' + str(50) + '.h5'
            weight_path = os.path.join(data_path, name)
            agent.load(weight_path)

            state_img = np.reshape(state_img, [1, state_size])
            action = agent(state_img)
            action_grid = A[action]
            wolf_next_state = physics(
                wolf_state, action_grid, env.is_state_valid)
            next_state_img = state_to_image_array(env,
                                                  [wolf_next_state], sheep_states, obstacle_states)
            plt.pause(0.1)
            plt.close('all')
            next_state_img = np.reshape(next_state_img, [1, state_size])

            state_value = agent.get_max_value(state_img)

            # value iteration
            transition_function = ft.partial(
                grid_transition, terminals=sheep_states, is_valid=env.is_state_valid)

            T = {s: {a: transition_function(s, a) for a in A} for s in S}

            grid_reward = ft.partial(
                get_reward, env=env, const=-1)
            func_lst = [grid_reward]
            reward_func = ft.partial(sum_rewards, func_lst=func_lst)
            R = {s: {a: reward_func(s, a) for a in A} for s in S}

            gamma = 0.9
            value_iteration = ValueIteration(
                gamma, epsilon=0.001, max_iter=100)
            V = value_iteration(S, A, T, R)

            state_value_ground_true = V[wolf_state]

            print state_value, state_value_ground_true

            wolf_state = wolf_next_state
            state_img = next_state_img
