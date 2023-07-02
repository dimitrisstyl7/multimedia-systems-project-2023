import numpy as np
from PIL import Image


def saveImage(image, name):
    """
    Save the image
    """
    image = Image.fromarray(image.astype(np.uint8))
    image.save('../auxiliary2023/dump/' + name)


def calculateErrorImage(image1, image2):
    """
    Calculate the error image of the current frame
    """
    return np.subtract(image1, image2)


def convertToUint8(image):
    """
    Convert the image to uint8
    """
    return image.astype(np.uint8)
