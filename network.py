
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Network(object):
    def __init__(self, name, num_actions, gamma=0.99, learning_rate=0.00025, momentum=0.95):
        """
        Args:
            includeLoss: boolean, if true, computation graph will include ops
                for computing loss and training
        """
        self.name = name

        with tf.variable_scope(self.name):
            # data layer
            self.input_state = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])

            # conv1 layer
            self.W_conv1 = weight_variable([8, 8, 4, 32])
            self.b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_state, self.W_conv1, strides=[1, 4, 4, 1], padding="SAME") + self.b_conv1)
            # image is now 21 x 21 x 32

            # conv2 layer
            self.W_conv2 = weight_variable([4, 4, 32, 64])
            self.b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, self.W_conv2, strides=[1, 2, 2, 1], padding="SAME") + self.b_conv2)
            # image is now 11 x 11 x 64

            # conv3 layer
            self.W_conv3 = weight_variable([3, 3, 64, 64])
            self.b_conv3 = bias_variable([64])
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, self.W_conv3, strides=[1, 1, 1, 1], padding="SAME") + self.b_conv3)
            # image is now 11 x 11 x 64

            # fc1 layer
            h_conv3_flat = tf.reshape(h_conv3, [-1, 11 * 11 * 64])
            self.W_fc1 = weight_variable([11 * 11 * 64, 512])
            self.b_fc1 = bias_variable([512])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

            # keep_prob = tf.placeholder(tf.float32)
            # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            # fc2 layer
            self.W_fc2 = weight_variable([512, num_actions])
            self.b_fc2 = bias_variable([num_actions])

            # distribution of actions
            self.q_values = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

            # selects the  single action with max value
            self.action = tf.argmax(self.q_values, 1)

            ##############
            # Loss layer #
            ##############

            # placeholder for rewards and actions, from transitions
            self.target_values = tf.placeholder(tf.float32, shape=[None])
            self.actions = tf.placeholder(tf.uint8, shape=[None])

            # compute the Q value of the actions taken in the first state of the transition
            observed_values = tf.reduce_sum(tf.mul(self.q_values, tf.one_hot(self.actions, num_actions)), axis=1)

            # compute the loss - uses Huber to clip gradient
            err = tf.reduce_mean(tf.sub(self.target_values, observed_values))
            self.loss = tf.select(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)

            ######################
            # Training optimizer #
            ######################

            self.train_step = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum).minimize(self.loss)
