
import numpy as np
from scipy.misc import imresize


def preprocess(raw_frame_deque):
    """
    Preprocessing images as described in the paper.
    1. take the max of the current frame and the previous frame
    2. convert to Y (luminescence) channel
    3. resize to a 84x84 image
    """
    assert len(raw_frame_deque) == 2
    frame0, frame1 = raw_frame_deque

    # # 1. max(prev_frame, cur_frame) because of flickering
    frame = np.maximum(frame0, frame1)

    # 2. convert to Y channel
    # formula from http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    # frame = 0.2126*red + 0.7152*green + 0.0722*blue

    frame = 0.299*red + 0.587*green + 0.114*blue

    # 3. resize to 84 x 84
    frame = imresize(frame, (84, 84))

    return frame
