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
from scipy.optimize import linprog
import random
from random import choice

class GridWorld():
    def __init__(self, name='', nx=None, ny=None):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.coordinates = tuple(it.product(range(self.nx), range(self.ny)))
        self.terminals = []
        self.obstacles = []
        self.features=co.OrderedDict()

    def add_terminals(self, terminals=[]):
        for t in terminals:
            self.terminals.append(t)

    def add_obstacles(self, obstacles=[]):
        for o in obstacles:
            self.obstacles.append(o)

    def add_feature_map(self, name, state_values, default=0):
        self.features[name] = {s:default for s in self.coordinates}
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
            return sum(map(lambda f:self.features[f][s_n], self.features))
        return sum(map(lambda f:self.features[f][s_n]*W[f], W.keys()))

    def draw_feature(self, ax, name, **kwargs):
        """kwargs passing to rvz.draw_2D_array"""
        I = dict_to_array(self.features[name])
        return draw_2D_array(I, ax, **kwargs)


    def draw_features_first_time(self, ax, features=[], colors={},
            masked_values={}, default_masked=0):
        assert set(features).issubset(set(self.features.keys()))

        if not features:
            features = self.features.keys()
        if len(features)>len(color_set):
            raise ValueError("there are %d features and only %d colors"
                             %(len(features), len(color_set)))


        free_color = list(filter(lambda c: c not in colors.values(),
                          color_set))
        colors.update({f:free_color.pop(0)
                       for f in features if f not in colors.keys()})
        masked_values.update({f:default_masked
                             for f in features if f not in masked_values.keys()})

        assert set(masked_values.keys()) == set(colors.keys()) == set(features)

        if not ax:
            fig, ax = plt.subplots(1, 1, tight_layout=True)

        def single_feature(ax, name):
            f_color = colors[name]
            masked_value = masked_values[name]

            return self.draw_feature(ax, name, f_color=f_color,
                                            masked_value=masked_value)

        ax_images = {f:single_feature(ax, f) for f in features}
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
        return {f:update_single_feature(f) for f in features}


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
            fig_name = os.path.join(save_to, str(self.name)+".png")
            plt.savefig(fig_name, dpi=200)
            if self.verbose>0:
                print ("saved %s"%fig_name)
        if show:
            plt.show()

        return ax, ax_images




def is_terminal(s):
    if s[:2]==s[2:]:
        return True

def grid_transition(s, a1, a2, is_valid=None):
    if s[:2]==s[2:]:
        return {s:1}

    s_n_1 = tuple(map(sum, zip(s[:2], a1)))
    s_n_2 = tuple(map(sum, zip(s[2:], a2)))

    if is_valid(s_n_1):
        if is_valid(s_n_2):
            s_n = s_n_1 + s_n_2
            return {s_n:1}
        else: 
            s_n = s_n_1 + s[2:]
            return {s_n:1}

    elif not is_valid(s_n_1):
        if is_valid(s_n_2):
            s_n = s[:2] + s_n_2
            return {s_n:1}

        else: return {s:1}


def reward_wolf(s):
    if s[:2]==s[2:]:
        return 500
    else:
        # return 100 / l2_norm(s[:2],s[2:])
        return 0

def reward_sheep(s):
    if s[:2]==s[2:]:
        return -500
    else:
        # return - l2_norm(s[:2],s[2:]) 
        return 0

def reward(s, a1, a2, is_invalid=None):
    s_n = physics(s, a1, a2, is_invalid)
    if s_n[:2]==s_n[2:]:
        return -500
    else:
        return - 500 / l2_norm(s_n[:2],s_n[2:])
        # return - 500 / l2_norm(s_n[:2],s_n[2:])

def trainsionFunctionToDict(S=(), A=(), tran_func=None):
    return {s:{a:tran_func(s, a) for a in A} for s in S}


def rewardFunctionToDict(S=(), A=(),T={}, reward_func=None):
    return {s:{a:{s_n:reward_func(s, a, s_n) for s_n in T[s][a]} for a in A} for s in S}


def policy_array_to_dict(PI_arr, S=(), A=()):
    return {s:{a:PI_arr[(s_i, a_i)] for (a_i, a) in enumerate(A)}
            for (s_i, s) in enumerate(S)}

def Q_array_to_dict(Q_arr, S=(), A1=(), A2=()):
    return {s:{(a1,a2):Q_arr[(s_i, a1_i, a2_i)] for (a1_i, a1) in enumerate(A1) for ((a2_i, a2)) in enumerate(A2) }  
            for (s_i, s) in enumerate(S)}

def V_array_to_dict(V_arr, S=()):
    return {s:V_arr[si] for (si,s) in enumerate(S)}


def sample_from_multinomial(prob_dict):
    keys, probabilities = zip(*prob_dict.iteritems())
    probabilities = np.asarray(probabilities)
    p_norm = probabilities/np.sum(probabilities)
    try:
        count  = np.random.multinomial(1, p_norm)
    except:
        count = np.random.multinomial(1, p_norm*0.999)
    index, = np.where(count==1)
    return keys[index[0]]


def validAction(s, A, is_invalid=None):
    valid_A = []
    for a in A:
        s_n = tuple(map(sum, zip(s, a)))
        if is_invalid(s_n):
            valid_A.append(a)
    return valid_A

def chooseAction1(s, S, A, pi, epsilon, env):
    valid_A = validAction(s[:2], A, is_invalid=env.is_state_valid)
    if np.random.rand() < epsilon:
        action = choice(valid_A)
    else:
        action = getOptimalAction(s, pi, valid_A)
    return action

def chooseAction2(s, S, A, pi, epsilon, env):
    valid_A = validAction(s[2:], A, is_invalid=env.is_state_valid)
    if np.random.rand() < epsilon:
        action = choice(valid_A)
    else:
        action = getOptimalAction(s, pi, valid_A)
    return action

def getOptimalAction(s, pi, A):
    pi = {s:{a: pi[s][a] for a in A } for s in S}
    max_actions = [key for key,val in pi[s].iteritems() if val == max(pi[s].values())]
    # action = max(pi[s].iteritems(), key=op.itemgetter(1))
    action = choice(max_actions)
    return action

def sample_multinomial_action_from_policy(state, PI):
    action_prob = PI.get(state)
    count = np.random.multinomial(1, action_prob.values())
    index, = np.where(count == 1)
    action = action_prob.keys()[index[0]]

    return action

class MinimaxQ():
    def __init__(self,numStates,numActionsWolf,numActionsSheep,gamma, decay, epsilon = 0.2, max_iter = 100, delta = 1e-4, alpha = 1):
        self.gamma = gamma
        self.decay = decay
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.delta = delta
        self.alpha = alpha
        self.numStates = numStates
        self.numActionsWolf = numActionsWolf
        self.numActionsSheep = numActionsSheep


    def updatePolicyWolf(self, state, pi, Q, V):
        c = np.zeros(self.numActionsWolf + 1)
        c[0] = -1
        A_ub = np.ones((self.numActionsSheep, self.numActionsWolf + 1))
        A_ub[:, 1:] = -Q[state].T
        b_ub = np.zeros(self.numActionsSheep)
        A_eq = np.ones((1, self.numActionsWolf + 1))
        A_eq[0, 0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * self.numActionsWolf

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        if res.success:
            pi[state] = res.x[1:]
        else:
            print("Alert : %s" % res.message)
            return V[state]

        return res.x[0]

    def updatePolicySheep(self, state, pi, Q, V):
        c = np.zeros(self.numActionsSheep + 1)
        c[0] = -1
        A_ub = np.ones((self.numActionsWolf, self.numActionsSheep + 1))
        A_ub[:, 1:] = -Q[state].T
        b_ub = np.zeros(self.numActionsWolf)
        A_eq = np.ones((1, self.numActionsSheep + 1))
        A_eq[0, 0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * self.numActionsSheep

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if res.success:
            pi[state] = res.x[1:]
        else:
            print("Alert : %s" % res.message)
            return V[state]

        return res.x[0]


    def __call__(self, S, A1, A2, T):
        gamma, epsilon, max_iter, alpha = self.gamma, self.epsilon, self.max_iter, self.alpha

        # V_A = {s: 1 for s in S}
        # Q_A = {s:{a1:{a2:1 for a2 in A2} for a1 in A1} for s in S}
        # pi_A = {s:{a:1  for a in A1} for s in S}

        V_A = np.ones(numStates)
        Q_A = np.ones((numStates, numActionsWolf, numActionsSheep))
        pi_A = np.ones((numStates, numActionsWolf)) / numActionsWolf

        # V_B = {s: 1 for s in S}
        # Q_B = co.defaultdict(ft.partial(np.random.rand, numStates, numActionsWolf, numActionsSheep))
        # pi_B = {s:{a:1  for a in A2} for s in S}

        V_B = np.ones(numStates)
        Q_B = np.ones((numStates, numActionsSheep, numActionsWolf))
        pi_B = np.ones((numStates, numActionsSheep)) / numActionsSheep

        start = (1,1,3,3)
        s = start

        for i in range(max_iter):
            if i % 100 == 0:
                print "iteration: ", i


            pi_A_dict = policy_array_to_dict(pi_A, S, A1)
            pi_B_dict = policy_array_to_dict(pi_B, S, A2)

            a1 = chooseAction1(s,S,A1,pi_A_dict,epsilon,env)
            a2 = chooseAction2(s,S,A2,pi_B_dict,epsilon,env)

            # s_n = sample_from_multinomial(T[s][(a1,a2)])
            s_n = physics(s,a1,a2,env.is_state_valid)

            si = S.index(s)
            a1i = A1.index(a1)
            a2i = A2.index(a2)
            s_ni = S.index(s_n)

            # reward_wolf = R1[s][a1]
            reward = reward_wolf(s_n)

            Q_A[si,a1i,a2i] = (1 - alpha) * Q_A[si,a1i,a2i] + \
                alpha * (reward + gamma * V_A[s_ni])
            V_A[si] = self.updatePolicyWolf(si, pi_A, Q_A, V_A)  

        #sheep 
            # Q_B = -Q_A
            # V_B = -V_A
            Q_B[si,a2i,a1i] = (1 - alpha) * Q_B[si,a2i,a1i] + \
                alpha * (-reward + gamma * V_B[s_ni])
            V_B[si] = self.updatePolicySheep(si, pi_B, Q_B, V_B)  

            alpha *= 10**(-2. / max_iter * 0.05)

            s = s_n
            
            if is_terminal(s):
                s = choice(S)

        return V_A, Q_A, pi_A, V_B, Q_B, pi_B


def V_dict_to_2Darray(V):
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

def softmax_policy(Q, temperature=10):
    na = Q.shape[-1] 
    q_exp = np.exp(Q/temperature - np.max(Q/temperature))
    norm = np.sum(q_exp, axis=1)
    prob = (q_exp/norm[:, np.newaxis])
    return prob

def softmax_epsilon_policy(Q, temperature=10, epsilon=0.1):
    na = Q.shape[-1]
    q_exp = np.exp(Q/temperature)
    norm = np.sum(q_exp, axis=1)
    prob = (q_exp/norm[:, np.newaxis])*(1-epsilon) + epsilon/na
    return prob

def softmax_epsilon_policy_stable(Q, temperature=10, epsilon=0.1):
    na = Q.shape[-1]
    q_exp = np.exp(Q/temperature - np.max(Q/temperature))
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
        s = sn


def pickle_dump_single_result(dirc="", prefix="result", name="", data=None):
    full_name = "_".join((prefix, name))+".pkl"
    path = os.path.join(dirc, full_name)
    pickle.dump(data, open(path, "wb"))
    print ("saving %s at %s"%(name, path))

def pickle_load_single_result(dirc="", prefix="result", name=""):
    full_name = "_".join((prefix, name))+".pkl"
    path = os.path.join(dirc, full_name)
    data = pickle.load(open(path, "rb"))
    print ("loading %s at %s"%(name, path))
    return data


def physics(s, a1, a2, is_valid=None):
    if s[:2]==s[2:]:
        return s

    s_n_1 = tuple(map(sum, zip(s[:2], a1)))
    s_n_2 = tuple(map(sum, zip(s[2:], a2)))

    if is_valid(s_n_1) and is_valid(s_n_2):
        s_n = s_n_1 + s_n_2
        return s_n

    elif is_valid(s_n_1) and not is_valid(s_n_2): 
        s_n = s_n_1 + s[2:]
        return s_n

    elif not is_valid(s_n_1) and is_valid(s_n_2): 
        s_n = s[:2] + s_n_2
        return s_n

    return s

if __name__ == '__main__':
    env = GridWorld("test", nx=5, ny=5)
    # obstacle_states = [(1,1)] # for 3*3
    obstacle_states = [(2,1), (2,3)] # for 5*5
    obstacles = {s:-10 for s in obstacle_states}
    env.add_feature_map("goal", obstacles, default=0)
    env.add_obstacles(obstacle_states)

    S = tuple(it.product(range(env.nx), range(env.ny),range(env.nx), range(env.ny)))

    A1 = ((1, 0), (0, 1), (-1, 0), (0, -1))
    A2 = ((1, 0), (0, 1), (-1, 0), (0, -1))

    transition_function = ft.partial(grid_transition, is_valid = env.is_state_valid)
    T = {s:{(a1,a2):transition_function(s, a1, a2) for a1 in A1 for a2 in A2 } for s in S}
    # T_arr = np.asarray([[[T[s][a].get(s_n, 0) for s_n in S] for a in A] for s in S])
    # print T


    # R1 = {s:{a1:reward_wolf(s, a1) for a1 in A1} for s in S}
    # R2 = {s:{a2:reward_sheep(s, a2) for a2 in A2} for s in S}


    gamma = 0.9

    numStates = len(S)
    numActionsWolf = len(A1)
    numActionsSheep = len(A2)

    # minimaxQ_agent = MinimaxQ(numStates, numActionsWolf, numActionsSheep, 
    #                             gamma, decay = 1e-4, epsilon = 0.1, max_iter = 100, delta = 1e-4)
    # V_A, Q_A, pi_A, V_B, Q_B, pi_B = minimaxQ_agent(S, A1, A2, T)

    # pi_A_dict = policy_array_to_dict(pi_A,S,A1)
    # pi_B_dict = policy_array_to_dict(pi_B,S,A1)
    # Q_A_dict = Q_array_to_dict(Q_A, S, A1, A2)
    # Q_B_dict = Q_array_to_dict(Q_B, S, A1, A2)
    # V_A_dict = V_array_to_dict(V_A, S)
    # V_B_dict = V_array_to_dict(V_B, S)

    # print Q_A_dict[(2,2,1,3)]
    # print pi_A_dict[(2,2,1,3)]

# save policy
    # module_path = os.path.dirname(os.path.abspath(__file__))
    # data_path = os.path.join(module_path, "minimax_policy")
    # prefix = "result_" + 'iteration' + str(minimaxQ_agent.max_iter)
    # pickle_dump_single_result(dirc=data_path, prefix=prefix, name="policyWolf", data=pi_A_dict)
    # pickle_dump_single_result(dirc=data_path, prefix=prefix, name="policySheep", data=pi_B_dict)
    # pickle_dump_single_result(dirc=data_path, prefix=prefix, name="QWolf", data=Q_A_dict)
    # pickle_dump_single_result(dirc=data_path, prefix=prefix, name="QSheep", data=Q_B_dict)

# load policy
    module_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(module_path, "minimax_policy")
    prefix = "result_" + "iteration10000"
    pi_A_dict = pickle_load_single_result(dirc=data_path, prefix=prefix, name="policyWolf")
    pi_B_dict = pickle_load_single_result(dirc=data_path, prefix=prefix, name="policySheep")
    
    # Q_A_dict = pickle_load_single_result(dirc=data_path, prefix=prefix, name="QWolf")
    # Q_B_dict = pickle_load_single_result(dirc=data_path, prefix=prefix, name="QSheep")

    # a1 = chooseAction((3,1,1,3),S,A1,pi_A_dict,0)
    # print a1

# animation
    obstacles = {s:-10 for s in obstacle_states}
    env.add_feature_map("obstacle", obstacles, default=0)

    s = (1,1,3,3)
    for i in range(1000000):

        wolf_states = [s[:2]]
        sheep_states = [s[2:]]

        wolf = {s:10 for s in wolf_states}
        sheep = {s:10 for s in sheep_states}

        env.add_feature_map("wolf", wolf, default=0)
        env.add_feature_map("sheep", sheep, default=0)

        ax, _ = env.draw(features=("wolf","sheep", "obstacle"), colors={'wolf':'r', 'sheep':'g', 'obstacle':'y'})
        # plt.pause(0.1)

        a1 = chooseAction1(s,S,A1,pi_A_dict,0,env)
        a2 = chooseAction2(s,S,A2,pi_B_dict,0,env)

        # s_n = sample_from_multinomial(T[s][(a1,a2)])

        s_n = physics(s, a1, a2, is_valid = env.is_state_valid)

        s = s_n

        if is_terminal(s):
            s = choice(S)

        print a1,reward_wolf(s)
        print a2,reward_sheep(s)
        print "================"

        # state to resized image

        fig = ax.get_figure()
        fig.canvas.draw()


        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        from PIL import Image

        pil_im = Image.fromarray(data)
        data = np.array(pil_im.resize((84, 84),3))

        print data.shape




