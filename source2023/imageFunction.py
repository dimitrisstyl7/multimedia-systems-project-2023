import numpy as np
from PIL import Image


def saveImage(image, name):
    image = Image.fromarray(image.astype(np.uint8))
    image.save(name)


def calculateErrorImage(image1, image2):
    return np.subtract(image1, image2)  # np.abs() returns the absolute value of the argument
