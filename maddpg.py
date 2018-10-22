import tensorflow as tf
import numpy as np

LAYER1_SIZE = 400
LAYER2_SIZE = 300
TAU = 0.001
LEARNING_RATE = 1e-3


class ActorNetwork:

    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        self.state_input, self.action_output, self.net, self.is_training = self.create_network(
            state_size, action_size)

		self.target_state_input, self.target_action_output, self.target_update, self.target_is_training = self.create_target_network(
		    state_dim, action_dim, self.net)

		self.sess.run(tf.initialize_all_variables())
        self.update_target()

    def create_network(self, state_size, action_size):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_size])
        is_training = tf.placeholder(tf.bool)

        W1 = self.variable([state_size, layer1_size], state_size)
        b1 = self.variable([layer1_size], state_size)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        W3 = tf.Variable(tf.random_uniform(
            [layer2_size, action_size], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([action_size], -3e-3, 3e-3))

        layer0_bn = self.batch_norm_layer(
            state_input, training_phase=is_training, scope_bn='batch_norm_0', activation=tf.identity)
        layer1 = tf.matmul(layer0_bn, W1) + b1
        layer1_bn = self.batch_norm_layer(
            layer1, training_phase=is_training, scope_bn='batch_norm_1', activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, W2) + b2
        layer2_bn = self.batch_norm_layer(
            layer2, training_phase=is_training, scope_bn='batch_norm_2', activation=tf.nn.relu)

        action_output = tf.tanh(tf.matmul(layer2_bn, W3) + b3)

        return state_input, action_output, [W1, b1, W2, b2, W3, b3], is_training

    def create_target_network(self, state_size, action_size, net):
        state_input = tf.placeholder("float", [None, state_size])
        is_training = tf.placeholder(tf.bool)
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer0_bn = self.batch_norm_layer(
            state_input, training_phase=is_training, scope_bn='target_batch_norm_0', activation=tf.identity)

        layer1 = tf.matmul(layer0_bn, target_net[0]) + target_net[1]
        layer1_bn = self.batch_norm_layer(
            layer1, training_phase=is_training, scope_bn='target_batch_norm_1', activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, target_net[2]) + target_net[3]
        layer2_bn = self.batch_norm_layer(
            layer2, training_phase=is_training, scope_bn='target_batch_norm_2', activation=tf.nn.relu)

        action_output = tf.tanh(
            tf.matmul(layer2_bn, target_net[4]) + target_net[5])

        return state_input, action_output, target_update, is_training

    def update_target(self):
    	self.sess.run(self.target_update)

     def __call__(self, obs):

        x = self.network_builder(obs)
        x = tf.layers.dense(x, self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        action_prob = tf.nn.tanh(x)

     	return action_prob

 class CriticNetwork:

    def __init__(self, sess, state_size, action_size):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size

        self.target_update = self.create_target_q_network(state_size,action_size,self.net)



    def create_network(self, state_size, action_size):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_size])
        is_training = tf.placeholder(tf.bool)

        W1 = self.variable([state_size, layer1_size], state_size)
        b1 = self.variable([layer1_size], state_size)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        W3 = tf.Variable(tf.random_uniform(
            [layer2_size, action_size], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([action_size], -3e-3, 3e-3))

        layer0_bn = self.batch_norm_layer(
            state_input, training_phase=is_training, scope_bn='batch_norm_0', activation=tf.identity)
        layer1 = tf.matmul(layer0_bn, W1) + b1
        layer1_bn = self.batch_norm_layer(
            layer1, training_phase=is_training, scope_bn='batch_norm_1', activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, W2) + b2
        layer2_bn = self.batch_norm_layer(
            layer2, training_phase=is_training, scope_bn='batch_norm_2', activation=tf.nn.relu)

        action_output = tf.tanh(tf.matmul(layer2_bn, W3) + b3)

        return state_input, action_output, [W1, b1, W2, b2, W3, b3], is_training

    def create_target_network(self, state_size, action_size, net):
        state_input = tf.placeholder("float", [None, state_size])
        is_training = tf.placeholder(tf.bool)
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer0_bn = self.batch_norm_layer(
            state_input, training_phase=is_training, scope_bn='target_batch_norm_0', activation=tf.identity)

        layer1 = tf.matmul(layer0_bn, target_net[0]) + target_net[1]
        layer1_bn = self.batch_norm_layer(
            layer1, training_phase=is_training, scope_bn='target_batch_norm_1', activation=tf.nn.relu)
        layer2 = tf.matmul(layer1_bn, target_net[2]) + target_net[3]
        layer2_bn = self.batch_norm_layer(
            layer2, training_phase=is_training, scope_bn='target_batch_norm_2', activation=tf.nn.relu)

        action_output = tf.tanh(
            tf.matmul(layer2_bn, target_net[4]) + target_net[5])

        return state_input, action_output, target_update, is_training

    def update_target(self):
    	self.sess.run(self.target_update)


    def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
		return tf.cond(training_phase, 
		lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
		updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
		lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
		updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

     def __call__(self, obs):

        x = self.network_builder(obs)
        x = tf.layers.dense(x, self.action_size, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        action_value = tf.nn.tanh(x)

     	return action_value



def train(sess, env, actor, critic, actor_noise):
	pass



class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, s2, done):
        experience = (s, a, r, s2, done)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        done_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch, done_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


''' 
1. critic network, target_network_for_Q
2. actor network, target_network_for_Policy
3. replay buffer
4. action_noise
5. optimize critic network
6. optimize actor network 
7. update target_network  
'''


if __name__ == '__main__':
	
    with tf.Session() as sess:

        # env = GridWorld("test", nx=5, ny=5)
        env = gym.make('Humanoid-v2')

        random_seed = 123

        num_of_agents = 2

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        state_size_all = state_size * num_of_agents
        action_size_all = action_size * num_of_agents

        actor1 = ActorNetwork(sess, state_size, action_size)
        actor2 = ActorNetwork(sess, state_size, action_size)

        critic = CriticNetwork(sess, state_size_all, action_size_all)

        replay_buffer = ReplayBuffer(buffer_size, random_seed)

        for episode in xrange(num_episodes):
            state = env.reset()
            obs1, obs2 = state

            for t in range(max_episode_length):
                action1 = actor1(obs1)
                action2 = actor2(obs2)
                action = action1 + action2

                next_state, reward, done, _ = env.step(action)

                replay_buffer.add((state, action, reward, next_state, done))

                train(sess, env, actor, critic, actor_noise)

                state = next_state

                if done:
                    break
