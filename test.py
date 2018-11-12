import unittest
import pdb
from dqn import *
from gridworld import *
from reward import *

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
    distance_reward, goal=sheep_states, dist_func=grid_dist, unit=1)

func_lst = [grid_reward, to_sheep_reward]
get_reward = ft.partial(sum_rewards, func_lst=func_lst)


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
        self.assertEqual(physics((2, 3), (1, 1), env.is_state_valid), (3, 4))
        self.assertNotEqual(
            physics((20, 20), (1, 1), env.is_state_valid), (21, 21))

    def test_state_to_image_array(self):
        self.assertEqual(state_to_image_array(env, image_size,
                                              [wolf_state], sheeps, obstacles).shape, (32, 32, 3))

        self.assertNotEqual(state_to_image_array(env, (21, 21),
                                                 [wolf_state], sheeps, obstacles).shape, (21, 21, 3))


if __name__ == '__main__':

    unittest.main()
