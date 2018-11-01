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


class GridWorld():
    def __init__(self, name='', nx=None, ny=None):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.coordinates = tuple(it.product(range(self.nx), range(self.ny)))
        self.terminals = []
        self.features=co.OrderedDict()

    def add_terminals(self, terminals=[]):
        for t in terminals:
            self.terminals.append(t)

    def add_feature_map(self, name, state_values, default=0):
        self.features[name] = {s:default for s in self.coordinates}
        self.features[name].update(state_values)
 

    def is_state_valid(self, state):
        if state[0] not in range(self.nx):
            return False
        if state[1] not in range(self.ny):
            return False
        return True

    def reward(self, s, a, s_n, W={}):
        if not W:
            return sum(map(lambda f:self.features[f][s_n], self.features))
        return sum(map(lambda f:self.features[f][s_n]*W[f], W.keys()))


    def draw(self, ax=None, ax_images={}, features=[], colors={},
                   masked_values={}, default_masked=0, show=True):

        new_features = [f for f in features if f not in ax_images.keys()]
        old_features = [f for f in features if f in ax_images.keys()]
        ax, new_ax_images = self.draw_features_first_time(ax, new_features,
                                    colors, masked_values, default_masked=0)
        old_ax_images = self.update_features_images(ax_images, old_features,
                                                    masked_values,
                                                    default_masked=0)
        ax_images.update(old_ax_images)
        ax_images.update(new_ax_images)


        return ax, ax_images



def T_dict(S=(), A=(), tran_func=None):
    return {s:{a:tran_func(s, a) for a in A} for s in S}


def R_dict(S=(), A=(),T={}, reward_func=None):
    return {s:{a:{s_n:reward_func(s, a, s_n) for s_n in T[s][a]} for a in A} for s in S}



def grid_transition(s, a, is_valid=None, terminals = ()):
    if s in terminals:
        return {s:1}
    s_n = tuple(map(sum, zip(s, a)))
    if is_valid(s_n):
        return {s_n:1}
    return {s:1}

def grid_transition_stochastic(s=(), a=(), is_valid=None, terminals=(), mode=0.9):
    if s in terminals:
        return {s:1}

    def apply_action(a, noise):
        return (s[0]+a[0]+noise[0], s[1]+a[1]+noise[1])

    s_n = apply_action(a, (0, 0))
    if not is_valid(s_n):
        return {s:1}

    #adding noise to next steps
    noise = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)]
    sn_iter = (apply_action(a, n) for n in noise)
    states = filter(is_valid, sn_iter)

    p_n = (1.0-mode)/len(states)

    prob = {s:p_n for s in states}
    prob[s_n] += mode
    return prob


def grid_obstacle_vanish_transition(s, a, is_valid=None, terminals = (), vanish_rate=0.1):
    s_n = tuple(map(sum, zip(s[:2], a)))

    if s[2:] == (0,0):
        if is_valid(s_n):
            return {(s_n+(0,0)):1}
        return {s:1}

    # s_n[2:] = (np.argmax(np.random.multinomial(1,[vanish_rate,1-vanish_rate])), np.argmax(np.random.multinomial(1,[vanish_rate,1-vanish_rate])))

    if is_valid(s_n):
        if s[2:] == (1,1):

            prob = {(s_n+(0,0)):vanish_rate*vanish_rate,
                    (s_n+(1,1)):(1-vanish_rate)*(1-vanish_rate),
                    (s_n+(0,1)):vanish_rate*(1-vanish_rate),
                    (s_n+(1,0)):(1-vanish_rate)*vanish_rate}

            return prob

        if s[2:] == (1,0):
            prob = {(s_n+(0,0)):vanish_rate,
                    (s_n+(1,0)):1-vanish_rate}

            return prob

        if s[2:] == (0,1):
            prob = {(s_n+(0,0)):vanish_rate,
                    (s_n+(0,1)):1-vanish_rate}

            return prob

    else:
        if s[2:] == (1,1):

            prob = {(s[:2]+(0,0)):vanish_rate*vanish_rate,
                    (s[:2]+(1,1)):(1-vanish_rate)*(1-vanish_rate),
                    (s[:2]+(0,1)):vanish_rate*(1-vanish_rate),
                    (s[:2]+(1,0)):(1-vanish_rate)*vanish_rate}

            return prob

        if s[2:] == (1,0):
            prob = {(s[:2]+(0,0)):vanish_rate,
                    (s[:2]+(1,0)):1-vanish_rate}

            return prob

        if s[2:] == (0,1):
            prob = {(s[:2]+(0,0)):vanish_rate,
                    (s[:2]+(0,1)):1-vanish_rate}

            return prob


def grid_reward(s, a, env=None, const=-1, is_terminal=None):
    return const + sum(map(lambda f:env.features[f][s], env.features))

def grid_obstacle_vanish_reward(s, a, env=None, const=-1, is_terminal=None, terminals = ()):
    if s[:2] == terminals[0] and s[2]==1:
        return const + sum(map(lambda f:env.features[f][s[:2]], env.features))
    if s[:2] == terminals[1] and s[3]==1:
        return const + sum(map(lambda f:env.features[f][s[:2]], env.features))
    else:
        return const


class ValueIteration():
    def __init__(self, gamma, epsilon = 0.001, max_iter = 100):
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
                                                for (s_n, p) in T[s][a].iteritems()]) for a in A ])

            delta = max(delta, abs(V_init[s] - V[s]))
            if delta < epsilon * (1 - gamma) / gamma:
                break
        return V

def dict_to_array(V):
    states, values = zip(*((s, v) for (s, v) in V.iteritems()))
    row_index, col_index = zip(*states)
    num_row = max(row_index)+1
    num_col = max(col_index)+1
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

def expected_value(R_arr, T_arr):
    ER_Q = np.sum(R_arr * T_arr, axis=2)
    return ER_Q

def V_to_Q(V, T=None, R=None, gamma=None):
    V_aug = V[np.newaxis, np.newaxis, :]
    return np.sum(T*(R+gamma*V_aug), axis=2)


def Q_from_V(s, a, T=None, R=None, V=None, gamma=None):
    return sum([p*(R[s][a] + gamma*V[s_n])
                for (s_n, p) in T[s][a].iteritems()])

def get_optimal_action(s, A=(), Q_func=None):
    Q = {a:Q_func(s, a) for a in A}
    return max(Q.iteritems(), key=op.itemgetter(1))[0]

def softmax_epislon_policy(Q, temperature=10, epsilon=0.1):
    na = Q.shape[-1]
    q_exp = np.exp(Q/temperature)
    norm = np.sum(q_exp, axis=1)
    prob = (q_exp/norm[:, np.newaxis])*(1-epsilon) + epsilon/na
    return prob

def action_sampler(s, PI=None, S=None, A=None):
    si = S.index(s)
    prob = PI[si]
    count = np.random.multinomial(1, prob.flatten())
    ai, = np.where(count==1)
    return A[ai[0]] # fix A[ai] to A[ai[0]] 


def follow_policy(s=(), state_sampler=None,  action_sampler=None,
                  is_invalid=None, viz_func=None, max_iter=1000):
    for i in range(max_iter):
        if viz_func:
            viz_func(s)
        a = action_sampler(s)
        sn = state_sampler(s, a)
        if is_invalid:
            if is_invalid(sn):
                print "terminal state! "
                break
        print s, a, sn
        s = sn


def pickle_dump_single_result(dirc="", prefix="result", name="", data=None):
    full_name = "_".join((prefix, name))+".pkl"
    path = os.path.join(dirc, full_name)
    pickle.dump(data, open(path, "wb"))
    print ("saving %s at %s"%(name, path))



if __name__ == '__main__':
    env = GridWorld("test", nx=2, ny=2)
    Q_merge = {}
    PI_merge = {}

    sheep_state = tuple(it.product(range(env.nx), range(env.ny)))
    sheep_states_two = list(it.product(sheep_state, sheep_state))

    for sheep_states in sheep_states_two:
        wolfs = {s:500 for s in sheep_states}
        env.add_feature_map("goal", wolfs, default=0)
        env.add_terminals(wolfs)

        S_2d = tuple(it.product(range(env.nx), range(env.ny)))
        # S = tuple(it.product(range(env.nx), range(env.ny),range(env.nx), range(env.ny)))
        S = tuple(it.product(range(env.nx), range(env.ny), range(2), range(2)))

        # A = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0), (1,1), (1,-1), (-1,1), (-1,-1))
        # A = tuple(it.product(range(-1, 2), range(-1, 2)))
        A = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))

        vanish_rate = 0.1
        transition_function = ft.partial(grid_obstacle_vanish_transition, terminals = sheep_states, is_valid = env.is_state_valid,vanish_rate=vanish_rate)

        T = {s:{a:transition_function(s, a) for a in A} for s in S}
        T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S] for a in A] for s in S])
        # print T

    # draw T
        # for ai, a in enumerate(A):
        #     fig, axes = plt.subplots(env.nx, env.ny, tight_layout=True)
        #     fig.set_size_inches(env.nx*3, env.ny*3, forward=True)
        #     draw_T(axes, T_arr, S ,a, ai)
     
        #     prefix = str(ai)
        #     name = "_".join((prefix, "transition.png"))
        #     module_path = os.path.dirname(os.path.abspath(__file__))
        #     figure_path = os.path.join(module_path, "figures")
        #     path = os.path.join(figure_path, name)
        #     print ("saving transition figure at %s"%path)
        #     plt.savefig(path, dpi=300)

        """set the reward func"""
        upper = np.array([env.nx, env.ny])
        lower = np.array([-1, -1])

        barrier_func = ft.partial(signod_barrier, c=0, m=50, s=1)
        barrier_punish = ft.partial(barrier_punish, barrier_func=barrier_func,upper=upper,lower=lower)
        # to_wolf_punish = ft.partial(distance_punish, goal=wolf_state, unit=1)
        to_wolf_punish = ft.partial(sigmoid_distance_punish, goal=sheep_states, unit=1)
        
        to_sheep_reward = ft.partial(distance_mean_reward, goal=sheep_states, unit=1)
        grid_reward = ft.partial(grid_obstacle_vanish_reward, env = env, const = -1, terminals = sheep_states)

        func_lst = [grid_reward]

        reward_func = ft.partial(sum_rewards, func_lst = func_lst)

        R = {s:{a:reward_func(s, a) for a in A} for s in S}
        R_arr = np.asarray([[[R[s][a] for s_n in S] for a in A] for s in S], dtype=float)

        # print R

    # draw R
        # for ai, a in enumerate(A):
        #     fig, axes = plt.subplots(env.nx, env.ny, tight_layout=True)
        #     fig.set_size_inches(env.nx*3, env.ny*3, forward=True)
        #     draw_R(axes, R_arr, S ,a, ai)
     
        #     prefix = str(ai)
        #     name = "_".join((prefix, "reward.png"))
        #     module_path = os.path.dirname(os.path.abspath(__file__))
        #     figure_path = os.path.join(module_path, "figures")
        #     path = os.path.join(figure_path, name)
        #     print ("saving transition figure at %s"%path)
        #     plt.savefig(path, dpi=300)

        gamma = 0.9

        value_iteration = ValueIteration(gamma, epsilon = 0.001, max_iter = 100)
        V = value_iteration(S, A, T, R)

        # print V

    # draw V
        # fig, ax = plt.subplots(1, 1, tight_layout=True)
        # fig.set_size_inches(env.nx*3, env.ny*3, forward=True)
        # draw_V(ax, V, S)
        # name = "sheep11_".join(('', "value.png"))
        # module_path = os.path.dirname(os.path.abspath(__file__))
        # figure_path = os.path.join(module_path, "figures")
        # path = os.path.join(figure_path, name)
        # print ("saving transition figure at %s"%path)
        # plt.savefig(path, dpi=300)


    # optimal action
        # Q_func = ft.partial(Q_from_V, T=T, R=R, V=V, gamma=gamma)
        # PI = {s:get_optimal_action(s, A=A, Q_func = Q_func) for s in S}
        # PI = {s:get_optimal_action(s, A=A, Q_func = Q_func) for s in S if s[2:]==(1,1)}
        

    # # softmax policy
        V_arr = V_dict_to_array(V)
        Q = V_to_Q(V = V_arr, T = T_arr, R = R_arr, gamma=gamma)
        # PI_func = ft.partial(softmax_epislon_policy, temperature=5,
        #                    epsilon=0.01)
        # PI = PI_func(Q.reshape(-1, len(A)))
        # # PI = {s:{a:PI[si, ai] for (ai, a) in enumerate(A)}
        # #          for (si, s) in enumerate(S)}

        # PI = {(s,sheep_states):{a:PI[si, ai] for (ai, a) in enumerate(A)}
        #          for (si, s) in enumerate(S)}


        Q_dict = {(s,sheep_states):{a:Q[si, ai] for (ai, a) in enumerate(A)}
                 for (si, s) in enumerate(S)}


        # action_sampler=ft.partial(action_sampler, PI=PI, S=S, A=A)
        # PI = {s:action_sampler(s) for s in S}

        #{s: [a, prob]}
        # PI = {s:[PI[s].items()] for s in S}

        # PI = {s:[PI[s].keys(),PI[s].values()] for s in S}

        # print PI

        # sheep_exist = (1,1)
        # V_2d = {s[:2]:v for s,v in V.items() if s[2:]==sheep_exist}
        # PI_2d = {s[:2]:v for s,v in PI.items() if s[2:]==sheep_exist}

        # print PI_2d

    # draw policy 
        # fig, ax = plt.subplots(1, 1, tight_layout=True)
        # fig.set_size_inches(env.nx*3, env.ny*3, forward=True)
        # # draw_policy_4d(ax ,PI = PI_2d, S=S_2d, V=V_2d)
        # draw_policy_4d_softmax(ax ,PI = PI_2d, S=S_2d, V=V_2d, A=A)
        # prefix = 'single_vanish_rate_'+ str(vanish_rate)+ str(sheep_exist) + str(sheep_states)+str(env.nx)
        # name = "wolf_".join((prefix, "policy.png"))
        # module_path = os.path.dirname(os.path.abspath(__file__))
        # figure_path = os.path.join(module_path, "figures4")
        # path = os.path.join(figure_path, name)
        # print ("saving policy figure at %s"%path)
        # plt.savefig(path, dpi=300)


    # save policy

        # module_path = os.path.dirname(os.path.abspath(__file__))
        # data_path = os.path.join(module_path, "result")
        # prefix = "result" + str(sheep_states)
        # pickle_dump_single_result(dirc=data_path, prefix=prefix, name="policy", data=PI)

        # PI_merge.update(PI)

        module_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(module_path, "Q_result")
        prefix = "result" + str(sheep_states)
        pickle_dump_single_result(dirc=data_path, prefix=prefix, name="value", data=Q_dict)
        
        Q_merge.update(Q_dict)

# save value
    print Q_dict

    module_path = os.path.dirname(os.path.abspath(__file__))
    figure_path = os.path.join(module_path, "Q_policy")
    prefix = 'vanish_rate_'+ str(vanish_rate) + str(sheep_states)
    pickle_dump_single_result(dirc=figure_path, prefix=prefix, name="value", data=Q_merge)

# save policy
    # module_path = os.path.dirname(os.path.abspath(__file__))
    # policy_path = os.path.join(module_path, "policy")
    # prefix = 'vanish_rate_'+ str(vanish_rate)
    # pickle_dump_single_result(dirc=policy_path, prefix=prefix, name="policy", data=PI_merge)




# animation

    # action_sampler = ft.partial(action_sampler, PI=PI_dict, S=S, A=A)

    # ax = env.draw()

    # viz_func = ft.partial(draw_episode, ax=ax, env = env, pause=0.2)

    # state_init = (1,1)
    # for i in range(10):
    #     follow_policy(s = state_init, PI, action_sampler = action_sampler, viz_func = viz_func)





