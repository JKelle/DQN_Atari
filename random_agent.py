
"""
This agent returns moves uniformly at random.
"""


import random


class RandomAgent(object):

  def __init__(self, legal_action_set):
    """
    Args:
      legal_action_set: list of legal actions (1D numpy array of int32)
    """
    self.legal_action_set = legal_action_set

  def getAction(self, state):
    """
    Maps a state to an action.

    Args:
      state: RGB image of game, as 3D numpy array
    """
    return random.choice(self.legal_action_set)
