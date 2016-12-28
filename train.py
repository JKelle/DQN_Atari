#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp


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

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)
    frame_history = deque([], frames_per_state)

    # epsilon-greedy parameters
    epsilon = 1.0
    epsilon_min = 0.2
    epsilon_delta = (1.0 - epsilon_min)/1000000

    counter = 0

    # TODO: fix this loop condition
    for episode in xrange(num_frames):

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
                action = random.choice(LEGAL_ACTIONS)
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
            replay_memory.append((cur_state, action, reward, next_state))

            # update epsilon
            epsilon -= epsilon_delta
            epsilon = max(epsilon, epsilon_min)

            cur_state = next_state
            counter += 1

            if len(replay_memory) < history_threshold:
                continue

            # sample transitions from replay_memory and peform SGD    
            transitions = random.sample(replay_memory, mini_batch_size)
            loss = agent.trainMiniBatch(transitions)
            if counter % 100 == 0:
                print "%i:\t%s\t%f\t%s minutes" % (counter, action, np.sqrt(loss.dot(loss)), (time.time() - START_TIME)/60)


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
