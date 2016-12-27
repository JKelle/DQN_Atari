
import tensorflow as tf


NUM_ACTIONS = 18


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


class Network(object):
	def __init__(self, name, gamma=0.99, learning_rate=0.00025, momentum=0.95):
		"""
		Args:
			includeLoss: boolean, if true, computatin graph will include ops
				for computing loss and training
		"""
		self.name = name

		with tf.variable_scope(self.name):
			# data layer
			self.input_state = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])

			# conv2 layer
			W_conv1 = weight_variable([8, 8, 4, 32])
			b_conv1 = bias_variable([32])
			h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_state, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1)
			# image is now 21 x 21 x 32

			# conv2 layer
			W_conv2 = weight_variable([4, 4, 32, 64])
			b_conv2 = bias_variable([64])
			h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)
			# image is now 11 x 11 x 64

			# conv3 layer
			W_conv3 = weight_variable([3, 3, 64, 64])
			b_conv3 = bias_variable([64])
			h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)
			# image is now 11 x 11 x 64

			# fc1 layer
			h_conv3_flat = tf.reshape(h_conv3, [-1, 11 * 11 * 64])
			W_fc1 = weight_variable([11 * 11 * 64, 512])
			b_fc1 = bias_variable([512])
			h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

			# keep_prob = tf.placeholder(tf.float32)
			# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

			# fc2 layer
			W_fc2 = weight_variable([512, NUM_ACTIONS])
			b_fc2 = bias_variable([NUM_ACTIONS])

			# distribution of actions
			self.q_values = tf.matmul(h_fc1, W_fc2) + b_fc2

			# selects the  single action with max value
			self.action = tf.argmax(self.q_values, 1)

			##############
			# Loss layer #
			##############

			# placeholder for rewards and actions, from transitions
			self.rewards = tf.placeholder(tf.float32, shape=[None])
			self.actions = tf.placeholder(tf.uint8, shape=[None])

			# placeholder for Q values from target network
			self.target_q_values = tf.placeholder(tf.float32, shape=[None, NUM_ACTIONS])

			# compute: r + gamma * max_{a'} Q_target(s, a')
			max_target_q_values = tf.reduce_max(self.target_q_values, axis=1)
			discounted_target_q_values = tf.scalar_mul(gamma, max_target_q_values)
			target_values = tf.add(self.rewards, discounted_target_q_values)

			# compute the Q value of the actions taken in the first state
			observed_values = tf.reduce_sum(tf.mul(self.q_values, tf.one_hot(self.actions, NUM_ACTIONS)))

			# compute the loss
			self.loss = tf.square(tf.sub(target_values, observed_values))

			######################
			# Training optimizer #
			######################

			self.train_step = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum).minimize(self.loss)
