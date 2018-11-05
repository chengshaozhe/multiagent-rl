import tensorflow as tf
# from tensorflow import keras
import numpy as np
import random
from collections import deque
import os
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from dqn import *
from value_iteration import *
from viz import *
from reward import *

if __name__ == '__main__':
    env = GridWorld("test", nx=11, ny=11)
    sheep_states = [(7, 7)]
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


# draw value map
    V_dqn = {}
    for wolf_state in S:
        state_img = state_to_image_array(env, image_size,
                                         [wolf_state], sheeps, obstacles)
        state_img = np.reshape(state_img, [1, state_size])
        V_dqn[wolf_state] = agent.get_state_value(state_img)
        print len(V_dqn)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    fig.set_size_inches(env.nx * 3, env.ny * 3, forward=True)

    draw_V(ax, V_dqn, S)
    prefix = 'v_v100' + \
        str(sheep_states) + str(env.nx)
    name = "wolf_".join((prefix, "policy.png"))
    module_path = os.path.dirname(os.path.abspath(__file__))
    figure_path = os.path.join(module_path, "figures")
    path = os.path.join(figure_path, name)
    print ("saving policy figure at %s" % path)
    plt.savefig(path, dpi=300)

# animation
    # done = False
    # num_opisodes = 1
    # for e in range(num_opisodes):
    #     wolf_state = random.choice(S)
    #     done = wolf_state in env.terminals
    #     while not done:
    #         state_img = state_to_image_array(env, image_size,
    #                                          [wolf_state], sheeps, obstacles)
    #         module_path = os.path.dirname(os.path.abspath(__file__))
    #         data_path = os.path.join(module_path, "linux/v100")
    #         name = str([(5, 5)]) + '_episode_' + str(130) + '.h5'
    #         weight_path = os.path.join(data_path, name)
    #         agent.load(weight_path)

    #         state_img = np.reshape(state_img, [1, state_size])

    #         action = agent(state_img)
    #         action_grid = A[action]
    #         wolf_next_state = physics(
    #             wolf_state, action_grid, env.is_state_valid)
    #         next_state_img = state_to_image_array(env, image_size,
    #                                               [wolf_next_state], sheeps, obstacles)
    #         plt.pause(0.1)
    #         plt.close('all')
    #         next_state_img = np.reshape(next_state_img, [1, state_size])

    #         state_value = agent.get_state_value(state_img)

    #         # value iteration
    #         transition_function = ft.partial(
    #             grid_transition, terminals=sheep_states, is_valid=env.is_state_valid)

    #         T = {s: {a: transition_function(s, a) for a in A} for s in S}

    #         grid_reward = ft.partial(
    #             grid_reward, env=env, const=-1)
    #         func_lst = [grid_reward]
    #         reward_func = ft.partial(sum_rewards, func_lst=func_lst)
    #         R = {s: {a: reward_func(s, a) for a in A} for s in S}

    #         gamma = 0.9
    #         value_iteration = ValueIteration(
    #             gamma, epsilon=0.001, max_iter=100)
    #         V = value_iteration(S, A, T, R)

    #         state_value_ground_true = V[wolf_state]

    #         print state_value, state_value_ground_true

    #         wolf_state = wolf_next_state
    #         state_img = next_state_img
