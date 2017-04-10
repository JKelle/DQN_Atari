#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp


import argparse
from collections import deque
import random
import sys

# import the Atari emulator
from ale_python_interface import ALEInterface
import numpy as np
import tensorflow as tf

# import agents
from dqn_agent import DQNAgent
from baseline_agent import BaselineAgent
from random_agent import RandomAgent
from utils import preprocess


ale = ALEInterface()

ale.setInt(b'random_seed', 123)

# prevent ALE from forcing repeated actions (default is 0.25)
ale.setFloat('repeat_action_probability', 0.0)

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = True
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


def play(agent, num_episodes=1, num_repeat_action=4, frames_per_state=4, epsilon=0.05, noop_max=30):
    """
    Actions:
         0 : no ball, no movement

         1 : starts ball, no movement

         2 : no ball, no movement
         3 : no ball, moves to the right
         4 : no ball, moves to the left
         5 : no ball, no movement
         6 : no ball, moves to the right
         7 : no ball, moves to the left
         8 : no ball, moves to the right
         9 : no ball, moves to the left

        10 : starts the ball, doesn't move
        11 : starts the ball, moves to the right
        12 : starts the ball, moves to the left
        13 : starts the ball, doesn't move
        14 : starts the ball, moves to the right
        15 : starts the ball, moves to the left
        16 : starts the ball, moves to the right
        17 : starts the ball, moves to the left
    """

    # Play 10 episodes ("episode" = one game)
    for episode_counter in range(num_episodes):
        ale.reset_game()

        # reset frame/state history
        preprocessed_frame_history = deque([], frames_per_state)
        raw_frame_deque = deque([], 2)

        is_terminal = False
        initial_lives = ale.lives()

        cur_state = startEpisode(
            noop_max, preprocessed_frame_history, raw_frame_deque,
            num_repeat_action, frames_per_state)
        assert cur_state.shape == (84, 84, 4)

        total_reward = 0

        # runs and episode
        while not is_terminal:
            _, reward, next_state = doTransition(
                ale, agent, cur_state, epsilon, num_repeat_action,
                preprocessed_frame_history, raw_frame_deque, initial_lives)

            total_reward += reward

            is_terminal = next_state is False

            cur_state = next_state

        print('Episode %d ended with score: %d' % (episode_counter, total_reward))


def main(sess):
    possible_agents = {
        "dqn": DQNAgent(sess, len(MINIMAL_ACTION_SET), 100000, 0.001),
        "random": RandomAgent(len(MINIMAL_ACTION_SET)),
        "baseline": BaselineAgent(),
    }

    parser = argparse.ArgumentParser(
        description="This program runs the Atari game Breakout with an AI agent.")

    parser.add_argument(
        '-a', '--agent',
        default='dqn',
        choices=possible_agents.keys(),
        dest='agent_type',
        help='specifies the type of agent used to play the game')

    args = parser.parse_args()

    agent = possible_agents[args.agent_type]

    play(agent, num_episodes=10, epsilon=0.0)



if __name__ == '__main__':
    with tf.Session() as sess:
        main(sess)
