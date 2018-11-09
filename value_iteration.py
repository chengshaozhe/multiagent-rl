import numpy as np
import collections as co
import functools as ft
import itertools as it
import operator as op
import matplotlib.pyplot as plt
import os
import pickle
from viz import *
from reward import *
import sys
sys.setrecursionlimit(2**30)
import pandas as pd
from gridworld import *


class ValueIteration():
    def __init__(self, gamma, epsilon=0.001, max_iter=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter

    def __call__(self, S, A, T, R):
        gamma, epsilon, max_iter = self.gamma, self.epsilon, self.max_iter
        V_init = {s: 0 for s in S}
        delta = 0
        for i in range(max_iter):
            V = V_init.copy()
            for s in S:
                V_init[s] = max([sum([p * (R[s][a] + gamma * V[s_n])
                                      for (s_n, p) in T[s][a].items()]) for a in A])

            delta = max(delta, abs(V_init[s] - V[s]))
            if delta < epsilon * (1 - gamma) / gamma:
                break
        return V


def dict_to_array(V):
    states, values = zip(*((s, v) for (s, v) in V.iteritems()))
    row_index, col_index = zip(*states)
    num_row = max(row_index) + 1
    num_col = max(col_index) + 1
    I = np.empty((num_row, num_col))
    I[row_index, col_index] = values
    return I


def V_dict_to_array(V):
    V_lst = [V.get(s) for s in S]
    V_arr = np.asarray(V_lst)
    return V_arr


def T_dict_to_array(T):
    T_lst = [[[T[s][a].get(s_n, 0) for s_n in S] for a in A] for s in S]
    T_arr = np.asarray(T_lst)
    return T_arr


def R_dict_to_array(R):
    R_lst = [[[R[s][a].get(s_n, 0) for s_n in S] for a in A] for s in S]
    R_arr = np.asarray(R_lst, dtype=float)
    return R_arr


def V_to_Q(V, T=None, R=None, gamma=None):
    V_aug = V[np.newaxis, np.newaxis, :]
    return np.sum(T * (R + gamma * V_aug), axis=2)


def Q_to_V(Q, ns=0, na=0):
    Q_2d = Q.reshape(ns, na)
    V = np.max(Q_2d, axis=1)  # axis=1 means maximize over columns (actions)
    return V


def Q_from_V(s, a, T=None, R=None, V=None, gamma=None):
    return sum([p * (R[s][a] + gamma * V[s_n])
                for (s_n, p) in T[s][a].iteritems()])


def get_optimal_action(s, A=(), Q_func=None):
    Q = {a: Q_func(s, a) for a in A}
    return max(Q.iteritems(), key=op.itemgetter(1))[0]


def softmax_epislon_policy(Q, temperature=10, epsilon=0.1):
    na = Q.shape[-1]
    q_exp = np.exp(Q / temperature)
    norm = np.sum(q_exp, axis=1)
    prob = (q_exp / norm[:, np.newaxis]) * (1 - epsilon) + epsilon / na
    return prob


def action_sampler(s, PI=None, S=None, A=None):
    si = S.index(s)
    prob = PI[si]
    count = np.random.multinomial(1, prob.flatten())
    ai, = np.where(count == 1)
    return A[ai[0]]  # fix A[ai] to A[ai[0]]


def follow_policy(s=(), state_sampler=None, action_sampler=None,
                  is_invalid=None, viz_func=None, max_iter=1000):
    for i in range(max_iter):
        if viz_func:
            viz_func(s)
        a = action_sampler(s)
        sn = state_sampler(s, a)
        if is_invalid:
            if is_invalid(sn):
                print ("terminal state! ")
                break
        print (s, a, sn)
        s = sn


if __name__ == '__main__':
    env = GridWorld("test", nx=21, ny=21)

    # sheep_states = [(3, 5), (3, 15)]
    sheep_states = [(5, 5)]

    wolfs = {s: 100 for s in sheep_states}
    env.add_feature_map("goal", wolfs, default=0)
    env.add_terminals(wolfs)

    S = tuple(it.product(range(env.nx), range(env.ny)))
    # A = tuple(it.product(range(-1, 2), range(-1, 2)))
    A = ((1, 0), (0, 1), (-1, 0), (0, -1))

    # transition_function = ft.partial(
    #     grid_transition_stochastic, terminals=sheep_states, is_valid=env.is_state_valid, mode=mode)

    # noise = 0.1
    transition_function = ft.partial(
        grid_transition, terminals=sheep_states, is_valid=env.is_state_valid)

    T = {s: {a: transition_function(s, a) for a in A} for s in S}
    T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S]
                         for a in A] for s in S])

    upper = np.array([env.nx, env.ny])
    lower = np.array([-1, -1])

    barrier_func = ft.partial(signod_barrier, c=0, m=50, s=1)
    barrier_punish = ft.partial(
        barrier_punish, barrier_func=barrier_func, upper=upper, lower=lower)
    # to_wolf_punish = ft.partial(distance_punish, goal=wolf_state, unit=1)
    to_wolf_punish = ft.partial(
        sigmoid_distance_punish, goal=sheep_states, unit=1)

    to_sheep_reward = ft.partial(
        distance_mean_reward, goal=sheep_states, unit=1)
    grid_reward = ft.partial(
        get_reward, env=env, const=-1)

    func_lst = [grid_reward]

    reward_func = ft.partial(sum_rewards, func_lst=func_lst)

    R = {s: {a: reward_func(s, a) for a in A} for s in S}
    R_arr = np.asarray([[[R[s][a] for s_n in S] for a in A]
                        for s in S], dtype=float)

    gamma = 0.9

    value_iteration = ValueIteration(gamma, epsilon=0.001, max_iter=100)
    V = value_iteration(S, A, T, R)

# # softmax policy
    # V_arr = V_dict_to_array(V)
    # Q = V_to_Q(V=V_arr, T=T_arr, R=R_arr, gamma=gamma)

    # Q_dict = {s: {a: Q[si, ai] for (ai, a) in enumerate(A)}
    #           for (si, s) in enumerate(S)}

    print (V)
