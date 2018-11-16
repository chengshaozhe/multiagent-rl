import unittest
from ddt import ddt, data, unpack

from dqn import *
from gridworld import *
from reward import *
from belief_dpn import belief_reward

env = GridWorld("test", nx=21, ny=21)
sheep_states = [(5, 5)]
obstacle_states = []
env.add_obstacles(obstacle_states)
env.add_terminals(sheep_states)

sheeps = {s: 500 for s in sheep_states}
obstacles = {s: -100 for s in obstacle_states}
env.add_feature_map("sheep", sheeps, default=0)
env.add_feature_map("obstacle", obstacles, default=0)

image_size = (32, 32)
wolf_state = (6, 5)

to_sheep_reward = ft.partial(
    distance_reward, goal=sheep_states[0], dist_func=grid_dist, unit=1)

func_lst = [grid_reward, to_sheep_reward]
get_reward = ft.partial(sum_rewards, func_lst=func_lst)


@ddt
class TestMDP(unittest.TestCase):

    def test_reward(self):
        self.assertLess(grid_reward((6, 5), (0, 1), env=env, const=-1),
                        grid_reward((5, 5), (0, 1), env=env, const=-1))
        self.assertEqual(grid_reward((6, 5), (0, 1), env=env, const=-1),
                         grid_reward((8, 5), (0, 1), env=env, const=-1))

        self.assertGreater(to_sheep_reward((6, 5), (0, 0)),
                           to_sheep_reward((8, 5), (0, 0)))

        self.assertEqual(to_sheep_reward((6, 6), (0, 0)), 100 / (2 + 1))

    def test_terminal(self):
        self.assertTrue((5, 5) in env.terminals)
        self.assertFalse((6, 5) in env.terminals)

    def test_physics(self):
        self.assertEqual(physics((2, 3), (1, 1), env), (3, 4))
        self.assertNotEqual(
            physics((20, 20), (1, 1), env), (21, 21))

    def test_state_to_image_array(self):
        self.assertEqual(state_to_image_array(env, (32, 32),
                                              [wolf_state], sheeps, obstacles).shape, (32, 32, 3))

        self.assertGreater((len(np.unique(state_to_image_array(env, (21, 21),
                                                               [wolf_state], sheeps, obstacles)))), 5)

    # @data((pd.DataFrame([np.random.normal(100, 1000, 100), np.random.uniform(-100, 200, 100)], columns=['a', 'b'])))
    @unpack
    def test_belief_reward(self):
        state1 = [(25, 25), (20, 20), (10, 35), [0, 0.2, 0.8]]
        state2 = [(25, 25), (20, 20), (10, 35), [0, 0.4, 0.6]]
        self.assertGreater(belief_reward(state1, (0, 0)),
                           belief_reward(state2, (0, 0)))

        state3 = [(25, 25), (24, 26), (22, 28), [0, 0.3, 0.7]]
        state4 = [(25, 25), (25, 28), (30, 25), [0, 0.4, 0.6]]
        self.assertEqual(belief_reward(state3, (0, 0)),
                         belief_reward(state4, (0, 0)))


if __name__ == '__main__':

    unittest.main()
