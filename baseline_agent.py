
"""
Manually coded agent.
The policy is to simply move the paddle's x coordinate to match
the ball's x coordinate.
"""

# hard-coded action codes for ALE
RIGHT = 11
LEFT = 12

WIDTH = 160
HEIGHT = 210

DEBUG = False


def getBallX(state):
    """
    Computes the x coordinate of the ball
    """
    region = state[93:189, 8:WIDTH-8, 0]
    nonzero_x_coords = region.nonzero()[1]
    if len(nonzero_x_coords) > 0:
        return nonzero_x_coords.mean()
    return -1


def getPaddleX(state):
    """
    Computes the x coordinate of the paddle.
    """
    region = state[190:191, 8:WIDTH-8, 0]
    nonzero_x_coords = region.nonzero()[1]
    assert len(nonzero_x_coords) > 0
    return nonzero_x_coords.mean()


class BaselineAgent(object):

    def getAction(self, state):
        """
        Maps state to an action.
        Move the paddle to be under the ball.

        Args:
            state: RGB image of game, as a numpy array

        Returns:
            an action ID, which is an int
        """
        ball_x = getBallX(state)
        paddle_x = getPaddleX(state)

        if ball_x == -1:
            # if ball not seen, move paddle to middle
            target_x = (WIDTH - 16)/2
        else:
            target_x = ball_x

        if target_x < paddle_x:
            action = LEFT
        else:
            action = RIGHT

        if DEBUG:
            print "ball_x =", ball_x
            print "paddle_x =", paddle_x
            print "target_x =", target_x
            print "action =", action
            raw_input()

        return action
