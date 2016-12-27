
import numpy as np
import tensorflow as tf

from network import Network


NUM_ACTIONS = 18


class DQNAgent(object):

	def __init__(self):
		# discount factor for future rewards
		self.gamma = 0.99

		# defines the convnet architecture in TensorFlow
		self.prediction_network = Network("prediction")
		self.target_network = Network("target")

	def getAction(self, state):
		"""
		Maps a single state to a single action.

		Args:
			state: a numpy array with shape
		"""
		# reshape [84, 84, 4] to be [1, 84, 84, 4]
		# and convert from uint8 to float32
		input_state = np.array([state], dtype=np.float32)
		
		# forward pass through the prediction network
		return self.prediction_network.action.eval(
			feed_dict={self.prediction_network.input_state:input_state})

	def trainMiniBatch(self, transitions):
		"""
		Args:
			transitions: list of (state1, action, reward, state2) tuples
				- state1: numpy array with shape (84, 84, 4)
				- action: scalar int in range [0, NUM_ACTIONS]
				- reward: scalar float in range [-1, 1]
				- state2: numpy array with shape (84, 84, 4)

		Returns:
			loss
		"""
		# reorganize data
		state1, actions, rewards, state2 = zip(*transitions)
		state1 = np.array(state1, dtype=np.float32)
		actions = np.array(actions, dtype=np.uint8)
		rewards = np.array(rewards, dtype=np.float32)
		state2 = np.array(state2, dtype=np.float32)

		# forward pass through the target network
		target_q_values = self.target_network.q_values.eval(
			feed_dict={self.target_network.input_state:state2})

		# TODO: gradient clipping

		feed_dict = {
			self.prediction_network.input_state: state1,
			self.prediction_network.actions: actions,
			self.prediction_network.rewards: rewards,
			self.prediction_network.target_q_values: target_q_values
		}

		self.prediction_network.train_step.run(feed_dict=feed_dict)

		return self.prediction_network.loss.eval(feed_dict=feed_dict)


if __name__ == '__main__':
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in range(20000):
	  batch = mnist.train.next_batch(50)
	  if i%100 == 0:
	    train_accuracy = accuracy.eval(feed_dict={
	        x:batch[0], y_: batch[1], keep_prob: 1.0})
	    print("step %d, training accuracy %g"%(i, train_accuracy))
	  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print("test accuracy %g"%accuracy.eval(feed_dict={
	    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))