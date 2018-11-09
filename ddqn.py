import tensorflow as tf
# from tensorflow import keras
import numpy as np
import random
from collections import deque
import functools as ft
import os
from PIL import Image
from viz import *
from reward import *
from keras import backend as K

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam


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

        if save_to:
            fig_name = os.path.join(save_to, str(self.name) + ".png")
            plt.savefig(fig_name, dpi=200)
            if self.verbose > 0:
                print ("saved %s" % fig_name)
        if show:
            plt.show()

        return ax, ax_images


def state_to_image_array(env, image_size, wolf_states, sheeps, obstacles):
    hit_wall_punish = -100
    wolf = {s: hit_wall_punish for s in wolf_states}
    env.add_feature_map("wolf", wolf, default=0)
    env.add_feature_map("sheep", sheeps, default=0)
    env.add_feature_map("obstacle", obstacles, default=0)

    ax, _ = env.draw(features=("wolf", "sheep", "obstacle"), colors={
                     'wolf': 'r', 'sheep': 'g', 'obstacle': 'y'})

    fig = ax.get_figure()
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(),
                          dtype=np.uint8, sep='')
    image_array = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    pil_im = Image.fromarray(image_array)

    image_array = np.array(pil_im.resize(image_size, 3))
    return image_array


def grid_reward(s, a, env=None, const=-10, is_terminal=None):
    return const + sum(map(lambda f: env.features[f][s], env.features))


def physics(s, a, is_valid=None):

    s_n = tuple(map(sum, zip(s, a)))

    if is_valid(s_n):
        return s_n
    return s


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            32, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(
            self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        actiton_values = self.model.predict(state)
        return np.argmax(actiton_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def __call__(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action_index = np.argmax(action_values[0])
        return action_index


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
    image_size = (84, 84)
    state_size = ft.reduce(lambda x, y: x * y, image_size) * 3
    agent = DQNAgent(state_size, action_size)

    batch_size = 16
    done = False
    num_opisodes = 1001

    for e in range(num_opisodes):
        wolf_state = random.choice(S)
        state_img = state_to_image_array(env, image_size,
                                         [wolf_state], sheeps, obstacles)
        for time in range(500):
            state_img = np.reshape(state_img, [1, state_size])
            action = agent.act(state_img)
            action_grid = A[action]
            wolf_next_state = physics(
                wolf_state, action_grid, env.is_state_valid)

            grid_reward = ft.partial(grid_reward, env=env, const=-10)
            to_sheep_reward = ft.partial(
                distance_reward, goal=sheep_states, unit=1)
            func_lst = [grid_reward, to_sheep_reward]

            get_reward = ft.partial(sum_rewards, func_lst=func_lst)

            reward = get_reward(wolf_state, action)
            done = wolf_next_state in env.terminals
            next_state_img = state_to_image_array(env, image_size,
                                                  [wolf_next_state], sheeps, obstacles)
            # plt.pause(0.1)
            plt.close('all')

            next_state_img = np.reshape(next_state_img, [1, state_size])

            agent.remember(state_img, action, reward, next_state_img, done)
            wolf_state = wolf_next_state
            state_img = next_state_img

            if done:
                agent.update_target_model()
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

                if time % 2 == 0:
                    print("episode: {}/{}, time: {}, reward: {:.4f}"
                          .format(e, num_opisodes, time, reward))

        if e % 10 == 0:
            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "save")
            name = str(sheep_states) + '_episode_' + str(e) + '.h5'
            weight_path = os.path.join(data_path, name)
            agent.save(weight_path)