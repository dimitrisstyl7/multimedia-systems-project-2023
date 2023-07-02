import numpy as np
from PIL import Image


def saveImage(image, name):
    image = Image.fromarray(image.astype(np.uint8))
    image.save('../auxiliary2023/dump/' + name)


def calculateErrorImage(image1, image2):
    return np.subtract(image1, image2)


def convertToUint8(image):
    return image.astype(np.uint8)
