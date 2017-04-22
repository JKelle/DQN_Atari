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

ale.setInt(b'random_seed', int(time.time()*1000) % 100000)

# prevent ALE from forcing repeated actions (default is 0.25)
ale.setFloat('repeat_action_probability', 0.0)

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

MINIMAL_ACTION_SET = ale.getMinimalActionSet()


def doTransition(ale, agent, cur_state, epsilon, num_skip_frames, preprocessed_frame_history, raw_frame_deque, initial_lives):
    # with probability epsilon, choose a random action
    if random.random() < epsilon:
        # take a random action
        action_index = random.choice(range(len(MINIMAL_ACTION_SET)))
    else:
        # choose action according the DQN policy
        action_index = agent.getAction(cur_state)

    action = MINIMAL_ACTION_SET[action_index]

    # repeat the action 4 times
    reward = 0
    for _ in range(num_skip_frames):
        reward += ale.act(action)
        raw_frame_deque.append(ale.getScreenRGB())
        if ale.game_over() or (ale.lives() < initial_lives):
            break

    # clip reward to range [-1, 1]
    reward = min(reward, 1)
    reward = max(reward, -1)

    # compute next state
    if ale.game_over() or (ale.lives() < initial_lives):
        next_state = False
    else:
        preprocessed_frame_history.append(preprocess(raw_frame_deque))

        # stack the most recent 4 frames
        next_state = np.stack(preprocessed_frame_history, axis=2)

    return action_index, reward, next_state


def startEpisode(noop_max, preprocessed_frame_history, raw_frame_deque, num_skip_frames, frames_per_state):
    """
    Do a random amount of noop actions to get starting state.
    """
    initial_lives = ale.lives()
    noop_counter = 0

    for _ in range(np.random.randint(4, noop_max + 1)):
        ale.act(0)
        raw_frame_deque.append(ale.getScreenRGB())
        noop_counter += 1

        if noop_counter % num_skip_frames == 0:
            preprocessed_frame_history.append(preprocess(raw_frame_deque))

    assert len(preprocessed_frame_history) > 0

    while len(preprocessed_frame_history) < frames_per_state:
        # copy the first processed frame until we have enough for a state
        preprocessed_frame_history.appendleft(preprocessed_frame_history[0])

    start_state = np.stack(preprocessed_frame_history, axis=2)

    is_terminal = ale.game_over() or (ale.lives() < initial_lives)
    assert not is_terminal

    return start_state


def main(replay_capacity=1000000, num_skip_frames=4,
         frames_per_state=4, mini_batch_size=32, history_threshold=50000,
         checkpoint_frequency=50000, target_network_update_frequency=10000,
         learning_rate=0.00025, noop_max=30):

    agent = DQNAgent(
        sess, len(MINIMAL_ACTION_SET), checkpoint_frequency, learning_rate=learning_rate)

    minibatch_counter = agent.getCounter()
    action_counter = 0
    episode_counter = 0

    # Initialize replay memory to capacity replay_capacity
    replay_memory = deque([], replay_capacity)

    # epsilon-greedy parameters
    epsilon_min = 0.1
    epsilon_delta = (1.0 - epsilon_min)/1000000

    # loaded from checkpoint - picking up where we left off
    epsilon = 1.0 - minibatch_counter*epsilon_delta
    epsilon = max(epsilon, epsilon_min)
    print "epsilon =", epsilon

    while True:
        episode_counter += 1
        ale.reset_game()

        # reset frame/state history
        preprocessed_frame_history = deque([], frames_per_state)
        raw_frame_deque = deque([], 2)

        is_terminal = False
        initial_lives = ale.lives()

        cur_state = startEpisode(
            noop_max, preprocessed_frame_history, raw_frame_deque,
            num_skip_frames, frames_per_state)

        # runs and episode
        while not is_terminal:

            action_index, reward, next_state = doTransition(
                ale, agent, cur_state, epsilon, num_skip_frames,
                preprocessed_frame_history, raw_frame_deque, initial_lives)

            replay_memory.append((cur_state, action_index, reward, next_state))

            is_terminal = next_state is False

            cur_state = next_state

            action_counter += 1

            if len(replay_memory) < history_threshold:
                # in 'burn-in' period; don't update epsilon, and don't do SGD
                if len(replay_memory) % 1000 == 0:
                    print "burn-in period. len(replay_memory) =", len(replay_memory)
                continue

            # update epsilon
            epsilon -= epsilon_delta
            epsilon = max(epsilon, epsilon_min)

            # update target network
            if action_counter % target_network_update_frequency == 0:
                print "updating target network ..."
                agent.updateTargetNetwork()

            # apply a minibatch SGD update after every 4 chosen actions
            if action_counter % 4 == 0:

                # sample uniformly from replay memory
                transitions = random.sample(replay_memory, mini_batch_size)
                loss = agent.trainMiniBatch(transitions)

                minibatch_counter += 1

                if minibatch_counter % 100 == 0:
                    print "\ttime =", (time.time() - START_TIME)/60
                    print "\tloss =", loss
                    print "\tepsilon =", epsilon
                    print "\taction_counter =", action_counter
                    print "\tepisode_counter =", episode_counter
                    print "\tminibatch_counter =", minibatch_counter
                    print "\tlen(replay_memory) =", len(replay_memory)
                    print "\tmost recent action =", MINIMAL_ACTION_SET[action_index]
                    print


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
