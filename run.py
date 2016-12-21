#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp

import sys
from random import randrange
from ale_python_interface import ALEInterface
from dqn_agent import DQNAgent
from baseline_agent import BaselineAgent
from random_agent import RandomAgent


ale = ALEInterface()

# Get & Set the desired settings
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

# Load the ROM file

if len(sys.argv) < 2:
    print("No ROM file specified. Using breakout.bin as default")
    rom_file = "roms/breakout.bin"
else:
    rom_file = str.encode(sys.argv[1])

ale.loadROM(rom_file)

# Get the list of legal actions
# myagent = DQNAgent(ale.getLegalActionSet())
# myagent = RandomAgent(ale.getLegalActionSet())
myagent = BaselineAgent()

# Play 10 episodes
for episode in range(10):
    total_reward = 0
    
    # play one game
    while not ale.game_over():
        # game state is just the pixels of the screen
        state = ale.getScreenRGB()
        
        # the agent maps states to actions
        action = myagent.getAction(state)
        
        # Apply an action and get the resulting reward
        reward = ale.act(action);
        total_reward += reward
    
    print('Episode %d ended with score: %d' % (episode, total_reward))
    
    # reset the game to run another episode
    ale.reset_game()
