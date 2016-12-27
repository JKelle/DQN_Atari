
import os

import numpy as np
import tensorflow as tf

from network import Network


NUM_ACTIONS = 18
CHECKPOINT_DIR = "checkpoints"


class DQNAgent(object):

	def __init__(self, sess, checkpoint_frequency):
		self.sess = sess

		# discount factor for future rewards
		self.gamma = 0.99
		self.target_network_update_frequency = 10000
		self.checkpoint_frequency = checkpoint_frequency
		self.update_counter = 0

		# defines the convnet architecture in TensorFlow
		self.prediction_network = Network("prediction")
		self.target_network = Network("target")

		self.saver = tf.train.Saver([
			self.target_network.W_conv1,
			self.target_network.W_conv2,
			self.target_network.W_conv3,
			self.target_network.W_fc1,
			self.target_network.W_fc2,
			self.target_network.b_conv1,
			self.target_network.b_conv2,
			self.target_network.b_conv3,
			self.target_network.b_fc1,
			self.target_network.b_fc2,
		])

		ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(sess, ckpt.model_checkpoint_path)
			self.update_counter = int(ckpt.model_checkpoint_path.split('-')[-1])
		else:
			print "WARNING: no checkpoint found"

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

		self.update_counter += 1
		if self.update_counter % self.target_network_update_frequency == 0:
			self._updateTargetNetwork()
		
		if self.update_counter % self.checkpoint_frequency == 0:
			print "checkpionting ... "
			self.saver.save(self.sess, os.path.join(CHECKPOINT_DIR, "model"), global_step=self.update_counter)

		return self.prediction_network.loss.eval(feed_dict=feed_dict)

	def _updateTargetNetwork(self):
		print "updating target network ..."
		self.sess.run(self.target_network.W_conv1.assign(self.prediction_network.W_conv1))
		self.sess.run(self.target_network.W_conv2.assign(self.prediction_network.W_conv2))
		self.sess.run(self.target_network.W_conv3.assign(self.prediction_network.W_conv3))
		self.sess.run(self.target_network.W_fc1.assign(self.prediction_network.W_fc1))
		self.sess.run(self.target_network.W_fc2.assign(self.prediction_network.W_fc2))
		self.sess.run(self.target_network.b_conv1.assign(self.prediction_network.b_conv1))
		self.sess.run(self.target_network.b_conv2.assign(self.prediction_network.b_conv2))
		self.sess.run(self.target_network.b_conv3.assign(self.prediction_network.b_conv3))
		self.sess.run(self.target_network.b_fc1.assign(self.prediction_network.b_fc1))
		self.sess.run(self.target_network.b_fc2.assign(self.prediction_network.b_fc2))
