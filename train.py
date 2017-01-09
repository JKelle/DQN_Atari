#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp

import sys
sys.path = [
    "/u/jkelle/.local/lib/python2.7/site-packages/Cython-0.21.1-py2.7-linux-x86_64.egg",
    "/u/jkelle/.local/lib/python2.7/site-packages/Mako-1.0.0-py2.7.egg",
    "/u/jkelle/.local/lib/python2.7/site-packages/pysynthetic-0.4.10-py2.7.egg",
    "/u/jkelle/.local/lib/python2.7/site-packages/PyContracts-1.6.6-py2.7.egg",
    "/v/filer4b/v20q001/jkelle/DQN_Atari",
    "/u/jkelle/vision/skpyutils",
    "/u/jkelle/vision/skvisutils",
    "/vision/vision_users/jkelle/tensorflow/lib/python2.7",
    "/vision/vision_users/jkelle/tensorflow/lib/python2.7/plat-x86_64-linux-gnu",
    "/vision/vision_users/jkelle/tensorflow/lib/python2.7/lib-tk",
    "/vision/vision_users/jkelle/tensorflow/lib/python2.7/lib-old",
    "/vision/vision_users/jkelle/tensorflow/lib/python2.7/lib-dynload",
    "/usr/lib/python2.7",
    "/usr/lib/python2.7/plat-x86_64-linux-gnu",
    "/usr/lib/python2.7/lib-tk",
    "/vision/vision_users/jkelle/tensorflow/local/lib/python2.7/site-packages",
    "/vision/vision_users/jkelle/tensorflow/lib/python2.7/site-packages",
    "/usr/lib/python2.7/dist-packages",
    "/u/jkelle/.local/lib/python2.7/site-packages",
    "/usr/local/lib/python2.7/site-packages",
    "/usr/local/lib/python2.7/dist-packages",
    "/usr/lib/python2.7/dist-packages/PILcompat",
    "/usr/lib/python2.7/dist-packages/gtk-2.0",
    "/usr/lib/pymodules/python2.7",
    "/usr/lib/python2.7/dist-packages/wx-2.8-gtk2-unicode",
]

import argparse
from collections import deque
import random
import time

from ale_python_interface import ALEInterface
import numpy as np
import tensorflow as tf

from dqn_agent import DQNAgent
from utils import preprocess


DEBUG = False

ale = ALEInterface()

LEGAL_ACTIONS = [1, 11, 12]

ale.setInt(b'random_seed', int(time.time()*1000))

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False
if USE_SDL:
    if sys.platform == 'darwin':
        import pygame
        pygame.init()
        ale.setBool('sound', False) # Sound doesn't work on OSX
    elif sys.platform.startswith('linux'):
        ale.setBool('sound', True)
    ale.setBool('display_screen', True)

rom_file = "roms/breakout.bin"
ale.loadROM(rom_file)


def main(num_frames=50000000, replay_capacity=1000000, num_skip_frames=4,
         frames_per_state=4, mini_batch_size=32, history_threshold=50000,
         checkpoint_frequency=1000, target_network_update_frequency=10000,
         learning_rate=0.00025):

    agent = DQNAgent(sess, checkpoint_frequency, target_network_update_frequency, learning_rate=learning_rate)

    counter = agent.getCounter()

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)
    frame_history = deque([], frames_per_state)

    # epsilon-greedy parameters
    epsilon_min = 0.1
    epsilon_delta = (1.0 - epsilon_min)/1000000

    # loaded from checkpoint - picking up where we left off
    epsilon = 1.0 - counter*epsilon_delta

    # clip epsilon
    epsilon = max(epsilon, epsilon_min)
    print "epsilon =", epsilon

    # TODO: fix this loop condition
    while True:

        ale.reset_game()

        # initialize the frame_history history by repeating the first frame
        cur_frame = ale.getScreenRGB()
        for _ in range(num_skip_frames):
            frame_history.append(preprocess(cur_frame))
        cur_state = np.stack(frame_history, axis=2)

        while not ale.game_over():

            # epsilon greedy
            if random.random() < epsilon:
                # take a random action
                action_index = random.choice(range(len(LEGAL_ACTIONS)))
            else:
                # choose action according the DQN policy
                action_index = agent.getAction(cur_state)

            action = LEGAL_ACTIONS[action_index]

            # repeat the action 4 times
            reward = 0
            for _ in range(num_skip_frames):
                reward += ale.act(action)

            # clip reward to range [-1, 1]
            reward = min(reward, 1)
            reward = max(reward, -1)

            # record new frame
            prev_frame = cur_frame
            cur_frame = ale.getScreenRGB()
            frame_history.append(preprocess(cur_frame, prev_frame))

            # stack the most recent 4 frames
            next_state = np.stack(frame_history, axis=2)

            # store transition in replay memory
            replay_memory.append((cur_state, action_index, reward, next_state))

            cur_state = next_state
            counter += 1

            if len(replay_memory) < history_threshold:
                if len(replay_memory) % 10000 == 0:
                    print "generated %d transitions" % len(replay_memory)
                continue

            # update epsilon
            epsilon -= epsilon_delta
            epsilon = max(epsilon, epsilon_min)

            # apply a minibatch SGD update after every 4 chosen actions
            if counter % 4 == 0:
                # sample uniformly from replay memory
                transitions = random.sample(replay_memory, mini_batch_size)
                loss = agent.trainMiniBatch(transitions)

            if counter % 100 == 0:
                print "%i:\t%s\t%f\t%s minutes" % (
                    counter,
                    action,
                    np.sqrt(loss.dot(loss)),
                    (time.time() - START_TIME)/60
                )


if __name__ == '__main__':
    START_TIME = time.time()
    with tf.Session() as sess:
        main(
            # num_frames=50000000,
            # replay_capacity=1000000,
            # num_skip_frames=4,
            # frames_per_state=4,
            # mini_batch_size=32,
            # history_threshold=500,
            # checkpoint_frequency=500,
            # target_network_update_frequency=1000
            # learning_rate=0.00010
        )
