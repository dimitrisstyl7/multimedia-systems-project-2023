import numpy as np


def calculateErrorImage(image1, image2):
    """
        Calculate the error image of the current frame
    """
    return np.subtract(image1, image2)
