#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp


import argparse
from collections import deque
import random

from ale_python_interface import ALEInterface
import numpy as np
import tensorflow as tf

from dqn_agent import DQNAgent
from utils import preprocess


DEBUG = False

ale = ALEInterface()

ale.setInt(b'random_seed', 123)

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


def main(num_frames=1000, replay_capacity=10000, num_repeat_action=4,
         frames_per_state=4, mini_batch_size=32, history_threshold=1000):
    
    agent = DQNAgent()
    sess.run(tf.global_variables_initializer())

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)
    recent_frames = deque([], frames_per_state)

    # epsilon-greedy parameters
    epsilon = 1.0
    epsilon_delta = (1.0 - 0.1)/1000000
    epsilon_min = 0.1

    # Initialize action-value function Q with random weights h
    # Initialize target action-value function Q^ with weights h2 5 h

    counter = 0

    for episode in xrange(num_frames):

        ale.reset_game()
        
        # for the first frame, just copy the same frame four times
        cur_frame = ale.getScreenRGB()
        cur_state = np.stack([preprocess(cur_frame)]*frames_per_state, axis=2)
        assert cur_state.shape == (84, 84, 4)

        while not ale.game_over():

            if DEBUG and counter % 100 == 0:
                print "epsilon =", epsilon

            # epsilon greedy
            if random.random() < epsilon:
                # take a random action
                if DEBUG and counter % 100 == 0:
                    print "about to pick random action"
                action = random.choice(ale.getLegalActionSet())
            else:
                # choose action according the DQN policy
                if DEBUG and counter % 100 == 0:
                    print "choosing on-policy action"
                action = agent.getAction(cur_state)

            if DEBUG and counter % 100 == 0:
                print "got action", action

            # take the action 4 times
            reward = 0
            for _ in range(num_repeat_action):
                prev_frame = cur_frame
                if DEBUG and counter % 100 == 0:
                    print "\t(taking action %i)" % action
                reward += ale.act(action)
                # TODO: handle "game over"
                cur_frame = ale.getScreenRGB()
                recent_frames.append(preprocess(cur_frame, prev_frame))

            # clip reward to range [-1, 1]
            reward = min(reward, 1)
            reward = max(reward, -1)

            if DEBUG and counter % 100 == 0:
                print "got reward", reward

            # stack the most recent 4 frames
            next_state = np.stack(recent_frames, axis=2)
            assert next_state.shape == (84, 84, 4)

            # store transition in replay memory
            replay_memory.append((cur_state, action, reward, next_state))
            cur_state = next_state
            counter += 1

            # update epsilon
            epsilon -= epsilon_delta
            epsilon = max(epsilon, epsilon_min)

            if len(replay_memory) > history_threshold:
                transitions = random.sample(replay_memory, mini_batch_size)
                loss = agent.trainMiniBatch(transitions)
                if counter % 100 == 0:
                    print
                    print counter
                    print loss


if __name__ == '__main__':
    with tf.Session() as sess:
        main()
