import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools as ft
import os
import csv
from PIL import Image
from viz import *
from reward import *

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


class GridWorld():
    def __init__(self, name='', nx=None, ny=None):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.coordinates = tuple(it.product(range(self.nx), range(self.ny)))
        self.terminals = []
        self.obstacles = []
        self.features = co.OrderedDict()

    def add_terminals(self, terminals=[]):
        for t in terminals:
            self.terminals.append(t)

    def add_obstacles(self, obstacles=[]):
        for o in obstacles:
            self.obstacles.append(o)

    def add_feature_map(self, name, state_values, default=0):
        self.features[name] = {s: default for s in self.coordinates}
        self.features[name].update(state_values)

    def is_state_valid(self, state):
        if state[0] not in range(self.nx):
            return False
        if state[1] not in range(self.ny):
            return False
        if state in self.obstacles:
            return False
        return True

    def reward(self, s, a, s_n, W={}):
        if not W:
            return sum(map(lambda f: self.features[f][s_n], self.features))
        return sum(map(lambda f: self.features[f][s_n] * W[f], W.keys()))

    def draw_feature(self, ax, name, **kwargs):
        I = dict_to_array(self.features[name])
        return draw_2D_array(I, ax, **kwargs)

    def draw_features_first_time(self, ax, features=[], colors={},
                                 masked_values={}, default_masked=0):
        assert set(features).issubset(set(self.features.keys()))

        if not features:
            features = self.features.keys()
        if len(features) > len(color_set):
            raise ValueError("there are %d features and only %d colors"
                             % (len(features), len(color_set)))

        free_color = list(filter(lambda c: c not in colors.values(),
                                 color_set))
        colors.update({f: free_color.pop(0)
                       for f in features if f not in colors.keys()})
        masked_values.update({f: default_masked
                              for f in features if f not in masked_values.keys()})

        assert set(masked_values.keys()) == set(colors.keys()) == set(features)

        if not ax:
            fig, ax = plt.subplots(1, 1, tight_layout=True)

        def single_feature(ax, name):
            f_color = colors[name]
            masked_value = masked_values[name]

            return self.draw_feature(ax, name, f_color=f_color,
                                     masked_value=masked_value)

        ax_images = {f: single_feature(ax, f) for f in features}
        return ax, ax_images

    def update_features_images(self, ax_images, features=[], masked_values={},
                               default_masked=0):
        def update_single_feature(name):
            try:
                masked_value = masked_values[name]
            except:
                masked_value = default_masked
            I = dict_to_array(self.features[name])
            return update_axes_image(ax_images[name], I, masked_value)
        return {f: update_single_feature(f) for f in features}

    def draw(self, ax=None, ax_images={}, features=[], colors={},
             masked_values={}, default_masked=0, show=False, save_to=''):

        plt.cla()
        if ax:
            ax.get_figure()
        new_features = [f for f in features if f not in ax_images.keys()]
        old_features = [f for f in features if f in ax_images.keys()]
        ax, new_ax_images = self.draw_features_first_time(ax, new_features,
                                                          colors, masked_values, default_masked=0)
        old_ax_images = self.update_features_images(ax_images, old_features,
                                                    masked_values,
                                                    default_masked=0)
        ax_images.update(old_ax_images)
        ax_images.update(new_ax_images)

        # if save_to:
        #     fig_name = os.path.join(save_to, str(self.name) + ".png")
        #     plt.savefig(fig_name, dpi=200)
        #     if self.verbose > 0:
        #         print ("saved %s" % fig_name)
        # if show:
        #     plt.show()

        return ax, ax_images


def state_to_image_array(env, image_size, wolf_states, sheeps, obstacles):
    wolf = {s: 1 for s in wolf_states}
    env.add_feature_map("wolf", wolf, default=0)
    env.add_feature_map("sheep", sheeps, default=0)
    env.add_feature_map("obstacle", obstacles, default=0)

    ax, _ = env.draw(features=("wolf", "sheep", "obstacle"), colors={
                     'wolf': 'r', 'sheep': 'g', 'obstacle': 'y'})

    fig = ax.get_figure()
    # fig.set_size_inches((image_size[0] / fig.dpi, image_size[1] / fig.dpi)) # direct resize
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(),
                          dtype=np.uint8, sep='')
    image_array = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# use PIL to resize
    pil_im = Image.fromarray(image_array)
    image_array = np.array(pil_im.resize(image_size[:2], 3))

    # print (image_array.shape)
    # print (len(np.unique(image_array)))
    return image_array


# def grid_reward(s, a, env=None, const=-10, is_terminal=None):
#     return const + sum(map(lambda f: env.features[f][s], env.features))


def grid_reward(s, a, env=None, const=-1):
    goal_reward = env.features['sheep'][s] if s in env.terminals else const
    obstacle_punish = env.features['obstacle'][s] if s in env.obstacles else 0
    return goal_reward + obstacle_punish


def physics(s, a, is_valid=None):
    s_n = tuple(map(sum, zip(s, a)))
    if is_valid(s_n):
        return s_n
    return s


class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_cnn()
        self.target_model = self._build_cnn()
        self.update_target_model()

    # def _build_model(self):
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(
    #         32, input_dim=self.state_size, activation='relu'))
    #     model.add(tf.keras.layers.Dense(32, activation='relu'))
    #     model.add(tf.keras.layers.Dense(
    #         self.action_size, activation='linear'))
    #     model.compile(loss='mse',
    #                   optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
    #     return model

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action = np.argmax(action_values[0])
        return action

    def get_state_value(self, state):
        action_values = self.model.predict(state)
        state_value = np.amax(action_values[0])
        return state_value

    def get_mean_action_values(self, state):
        action_values = self.model.predict(state)
        state_value = np.mean(action_values[0])
        return state_value

    def get_Q(self, state):
        action_values = self.model.predict(state)
        return action_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

                states.append(state[0])
                targets.append(target[0])

        states_mb = np.array(states)
        targets_mb = np.array(targets)
        return states_mb, targets_mb

    def train(self, states_mb, targets_mb):
        history = self.model.fit(states_mb, targets_mb, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss']

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def __call__(self, state_img):
        action_values = self.model.predict(state_img)
        action_index_max = np.argmax(action_values[0])
        return action_index_max


def log_results(filename, loss_log):
    with open('results/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


if __name__ == '__main__':
    env = GridWorld("test", nx=21, ny=21)
    sheep_states = [(5, 5)]
    obstacle_states = []
    env.add_obstacles(obstacle_states)
    env.add_terminals(sheep_states)

    sheeps = {s: 500 for s in sheep_states}
    obstacles = {s: -100 for s in obstacle_states}

    S = tuple(it.product(range(env.nx), range(env.ny)))
    A = ((1, 0), (0, 1), (-1, 0), (0, -1))
    action_size = len(A)
    image_size = (21, 21, 3)
    state_size = ft.reduce(lambda x, y: x * y, image_size) * 3

    agent = DQNAgent(image_size, action_size)
    # agent.load("./save/[(5, 5)]_episode_120.h5")
    loss_log = []

    batch_size = 32
    replay_start_size = 1000
    num_opisodes = 1001
    done = False

    for e in range(num_opisodes):
        wolf_state = random.choice(S)
        state_img = state_to_image_array(env, image_size,
                                         [wolf_state], sheeps, obstacles)
        state_img = np.reshape(
            state_img, [1, image_size[0], image_size[1], image_size[2]])
        for time in range(1000):

            action = agent.act(state_img)
            action_grid = A[action]

            wolf_next_state = physics(
                wolf_state, action_grid, env.is_state_valid)

            grid_reward = ft.partial(grid_reward, env=env, const=-1)
            to_sheep_reward = ft.partial(
                distance_reward, goal=sheep_states, dist_func=grid_dist, unit=1)
            func_lst = [grid_reward, to_sheep_reward]
            get_reward = ft.partial(sum_rewards, func_lst=func_lst)

            reward = get_reward(wolf_state, action)

            done = wolf_state in env.terminals
            next_state_img = state_to_image_array(env, image_size,
                                                  [wolf_next_state], sheeps, obstacles)
            # plt.pause(0.1)
            plt.close('all')

            next_state_img = np.reshape(
                next_state_img, [1, image_size[0], image_size[1], 3])

            agent.remember(state_img, action, reward, next_state_img, done)
            wolf_state = wolf_next_state
            state_img = next_state_img

            if done:
                agent.update_target_model()
                break

            if len(agent.memory) > replay_start_size:
                states_mb, targets_mb = agent.replay(batch_size)
                loss = agent.train(states_mb, targets_mb)

                if time % 10 == 0:

                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(e, num_opisodes, time, loss[0]))

                    loss_log.append(loss)

        if e % 10 == 0:
            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "save")
            name = str(sheep_states) + '_episode_' + str(e) + '.h5'
            weight_path = os.path.join(data_path, name)
            agent.save(weight_path)

            filename = str(image_size) + '-' + \
                str(batch_size) + 'episode-' + str(e)
            log_results(filename, loss_log)
            loss_log = []
