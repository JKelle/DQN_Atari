
import os

import numpy as np
import tensorflow as tf

from network import Network


CHECKPOINT_DIR = "checkpointReducedActionSet"
CHECKPOINT_DIR = "/vision/vision_users/jkelle/checkpoint2"


class DQNAgent(object):

    def __init__(self, sess, num_actions, checkpoint_frequency, learning_rate):
        self.sess = sess

        # discount factor for future rewards
        self.gamma = 0.99
        self.checkpoint_frequency = checkpoint_frequency
        self.counter = 0

        # defines the convnet architecture in TensorFlow
        self.prediction_network = Network("prediction", num_actions, learning_rate=learning_rate)
        self.target_network = Network("target", num_actions, learning_rate=learning_rate)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(
            [
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
                self.prediction_network.W_conv1,
                self.prediction_network.W_conv2,
                self.prediction_network.W_conv3,
                self.prediction_network.W_fc1,
                self.prediction_network.W_fc2,
                self.prediction_network.b_conv1,
                self.prediction_network.b_conv2,
                self.prediction_network.b_conv3,
                self.prediction_network.b_fc1,
                self.prediction_network.b_fc2,
            ],
            max_to_keep=2
        )

        self.update_target_network = [
            self.target_network.W_conv1.assign(self.prediction_network.W_conv1),
            self.target_network.W_conv2.assign(self.prediction_network.W_conv2),
            self.target_network.W_conv3.assign(self.prediction_network.W_conv3),
            self.target_network.W_fc1.assign(self.prediction_network.W_fc1),
            self.target_network.W_fc2.assign(self.prediction_network.W_fc2),
            self.target_network.b_conv1.assign(self.prediction_network.b_conv1),
            self.target_network.b_conv2.assign(self.prediction_network.b_conv2),
            self.target_network.b_conv3.assign(self.prediction_network.b_conv3),
            self.target_network.b_fc1.assign(self.prediction_network.b_fc1),
            self.target_network.b_fc2.assign(self.prediction_network.b_fc2),
        ]

        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print "loading checkpoint parameters ..."
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.counter = int(ckpt.model_checkpoint_path.split('-')[-1])
            print "done"
        else:
            print "WARNING: no checkpoint found. Starting weights from scratch."

        # self._assertSameWeights()

    def getCounter(self):
        return self.counter

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
            feed_dict={self.prediction_network.input_state: input_state})[0]

    def _computeTargetValues(self, rewards, next_states):
        """
        Used by trainMiniBatch().

        Args:
            rewards: list of scalars in the range [-1, 1]
            next_states: list of "state2" arrays, or False if the transition
                lead to a terminal state?

        Returns:
            numpy array of target values ("labels") used to compute loss
        """
        labels = []

        for reward, state2 in zip(rewards, next_states):
            if state2 is False:
                labels.append(reward)
            else:
                # forward pass through the target network
                target_q_values = self.target_network.q_values.eval(
                    feed_dict={self.target_network.input_state:[state2]})
                labels.append(reward + self.gamma*target_q_values.max())

        # cast to numpy array
        return np.array(labels, np.float32)

    def trainMiniBatch(self, transitions):
        """
        Args:
            transitions: list of (state1, action, reward, state2) tuples
                - state1: numpy array with shape (84, 84, 4)
                - action: scalar int in range [0, NUM_ACTIONS]
                - reward: scalar float in range [-1, 1]
                - state2: numpy array with shape (84, 84, 4), or False if this
                    transition lead to a terminal state

        Returns:
            loss (scalar)
        """
        # reorganize data
        state1, actions, rewards, state2 = zip(*transitions)
        state1 = np.array(state1, dtype=np.float32)
        actions = np.array(actions, dtype=np.uint8)

        # compute target values ("labels")
        target_values = self._computeTargetValues(rewards, state2)

        feed_dict = {
            self.prediction_network.input_state: state1,
            self.prediction_network.actions: actions,
            self.prediction_network.target_values: target_values
        }

        self.prediction_network.train_step.run(feed_dict=feed_dict)

        self.counter += 1

        # checkpoint (save weights to disk)
        if self.counter % self.checkpoint_frequency == 0:
            print "checkpointing %d ... " % self.counter
            self.saver.save(
                self.sess,
                os.path.join(CHECKPOINT_DIR, "model"),
                global_step=self.counter
            )

        return self.prediction_network.loss.eval(feed_dict=feed_dict)

    def updateTargetNetwork(self):
        self.sess.run(self.update_target_network)
        # self._assertSameWeights()

    def _assertSameWeights(self):
        target_weights = self.sess.run([
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
        prediction_weights = self.sess.run([
            self.prediction_network.W_conv1,
            self.prediction_network.W_conv2,
            self.prediction_network.W_conv3,
            self.prediction_network.W_fc1,
            self.prediction_network.W_fc2,
            self.prediction_network.b_conv1,
            self.prediction_network.b_conv2,
            self.prediction_network.b_conv3,
            self.prediction_network.b_fc1,
            self.prediction_network.b_fc2,
        ])

        for target_tensor, prediction_tensor in zip(target_weights, prediction_weights):
            assert (target_tensor == prediction_tensor).all()

        print "True"
