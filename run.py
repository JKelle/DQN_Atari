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

LEGAL_ACTIONS = [1, 11, 12]

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

num_episodes=1
num_repeat_action=4
frames_per_state=4
def play(agent, num_episodes=1, num_repeat_action=4, frames_per_state=4, epsilon=0.05):
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
    recent_frames = deque([], frames_per_state)

    # Play 10 episodes ("episode" = one game)
    for episode in xrange(num_episodes):
        total_reward = 0
        
        # reset the game
        ale.reset_game()
        is_new_game = True
        
        # for the first frame, just copy the same frame four times
        cur_frame = ale.getScreenRGB()
        cur_state = np.stack([preprocess(cur_frame)]*frames_per_state, axis=2)
        assert cur_state.shape == (84, 84, 4)
        
        # play one game
        while not ale.game_over():
            
            # # epsilon greedy
            # if is_new_game:
            #     action = 1  # serve the ball, don't move the paddle
            #     is_new_game = False
            if random.random() < epsilon:
                # take a random action
                action = random.choice(LEGAL_ACTIONS)
                # print "chose action", action, "*"
            else:
                # choose action according the DQN policy
                action_index = agent.getAction(cur_state)
                action = LEGAL_ACTIONS[action_index]
                # print "chose action", action


            # take the action 4 times
            reward = 0
            for _ in range(num_repeat_action):
                prev_frame = cur_frame
                reward += ale.act(action)
                cur_frame = ale.getScreenRGB()
                recent_frames.append(preprocess(cur_frame, prev_frame))

            # clip reward to range [-1, 1]
            reward = min(reward, 1)
            reward = max(reward, -1)
            total_reward += reward

            # stack the most recent 4 frames
            cur_state = np.stack(recent_frames, axis=2)
        
        print('Episode %d ended with score: %d' % (episode, total_reward))
        
        # reset the game to run another episode
        ale.reset_game()


def main(sess):
    possible_agents = {
        "dqn": DQNAgent(sess, 100000, 100000, 0.00001),
        "random": RandomAgent(LEGAL_ACTIONS),
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