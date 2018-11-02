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
             masked_values={}, default_masked=0, show=True, save_to=''):

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


def state_to_image_array(env, wolf_states, sheep_states, obstacle_states):
    wolf = {s: -100 for s in wolf_states}
    sheep = {s: 500 for s in sheep_states}
    obstacles = {s: -100 for s in obstacle_states}
    env.add_feature_map("wolf", wolf, default=0)
    env.add_feature_map("sheep", sheep, default=0)
    env.add_feature_map("obstacle", obstacles, default=0)

    ax, _ = env.draw(features=("wolf", "sheep", "obstacle"), colors={
                     'wolf': 'r', 'sheep': 'g', 'obstacle': 'y'})
    fig = ax.get_figure()
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(),
                          dtype=np.uint8, sep='')
    image_array = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    pil_im = Image.fromarray(image_array)

    # image_size = ((84, 84), 3)
    image_array = np.array(pil_im.resize((84, 84), 3))

    print image_array.shape
    return image_array


def get_reward(s, a, env=None, const=-10, is_terminal=None):
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
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(
            targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

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

    obstacles = {s: -100 for s in obstacle_states}
    sheep = {s: 500 for s in sheep_states}
    env.add_feature_map("sheep", sheep, default=0)

    S = tuple(it.product(range(env.nx), range(env.ny)))
    A = ((1, 0), (0, 1), (-1, 0), (0, -1))
    action_size = len(A)

    num_opisodes = 100
    batch_size = 256
    for e in range(num_opisodes):
        wolf_state = random.choice(S)
        state_img = state_to_image_array(env,
                                         [wolf_state], sheep_states, obstacle_states)
        # plt.pause(0.1)
        state_size = state_img.flatten().shape[0]

        agent = DQNAgent(state_size, action_size)
        for time in range(1000):
            state_img = np.reshape(state_img, [1, state_size])
            action = agent.act(state_img)
            action_grid = A[action]
            wolf_next_state = physics(
                wolf_state, action_grid, env.is_state_valid)

            reward = get_reward(wolf_next_state, action, env)
            done = wolf_next_state in env.terminals
            next_state_img = state_to_image_array(env,
                                                  [wolf_next_state], sheep_states, obstacle_states)
            plt.pause(0.1)
            plt.close('all')

            next_state_img = np.reshape(next_state_img, [1, state_size])

            agent.remember(state_img, action, reward, next_state_img, done)
            wolf_state = wolf_next_state
            state_img = next_state_img

            if done:
                break

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)

                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(e, num_opisodes, time, loss))

        if e % 10 == 0:
            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "save")
            name = str(sheep_states) + str(e) + '.h5'
            weight_path = os.path.join(data_path, name)
            agent.save(weight_path)
