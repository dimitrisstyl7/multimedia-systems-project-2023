import cv2
import numpy as np


def motionEstimation(prev_frame, curr_frame, width, height):
    """
    Perform motion estimation
    """
    # The motion vectors
    motion_vectors = []

    return motion_vectors


def fullSearch(curr_frame_macroblocks, prev_frame_macroblocks, k):
    pass


def getMacroblockSize(level):
    if level == 0:
        return 64
    elif level == 1:
        return 32
    return 16  # level 2


def getSearchRadius(level):
    if level == 0:
        return 32
    elif level == 1:
        return 16
    return 8  # level 2


def getWidthAndHeight(level, width, height):
    if level == 0:
        return width, height
    elif level == 1:
        return width // 2, height // 2
    return width // 4, height // 4  # level 2


def hierarchicalSearch(frames, width, height):
    """

    """
    motion_vectors = []
    for P in range(1, len(frames)):
        prev_frame = frames[P - 1]
        curr_frame = frames[P]

        # subsample previous and current frames in 3 levels
        curr_frame_levels = [curr_frame,
                             cv2.resize(curr_frame, (width / 2, height / 2), interpolation=cv2.INTER_LINEAR),
                             cv2.resize(curr_frame, (width / 4, height / 4), interpolation=cv2.INTER_LINEAR)]

        prev_frame_levels = [prev_frame,
                             cv2.resize(prev_frame, (width / 2, height / 2), interpolation=cv2.INTER_LINEAR),
                             cv2.resize(prev_frame, (width / 4, height / 4), interpolation=cv2.INTER_LINEAR)]

        # search macroblocks - motion vectors
        levels = len(curr_frame_levels)
        for level in range(0, levels):
            macroblock_size = getMacroblockSize(level)
            k = getSearchRadius(level)  # search radius (in pixels)
            width, height = getWidthAndHeight(level, width, height)
            executeLevel(curr_frame_levels[level], prev_frame_levels[level], macroblock_size, k, width, height)

        motion_vectors.append(motionEstimation(prev_frame, curr_frame, width, height))


def divideFrameIntoMacroblocks(frame, macroblock_size, width, height):
    macroblocks = []
    for row in range(0, height, macroblock_size):
        for col in range(0, width, macroblock_size):
            macroblocks.append(frame[row: row + macroblock_size, col: col + macroblock_size])
    return np.array(macroblocks)


def getSADErrorValue(curr_frame_macroblocks, prev_frame_macroblocks, k):
    """
        Sum of Absolute Differences (SAD)
    """
    sad = 0
    for i in range(k):
        for j in range(k):
            sad += abs(int(curr_frame_macroblocks[i, j]) - int(prev_frame_macroblocks[i, j]))
    return sad


def executeLevel(curr_frame, prev_frame, macroblock_size, k, width, height):
    curr_frame_macroblocks = divideFrameIntoMacroblocks(curr_frame, macroblock_size, width, height)
    prev_frame_macroblocks = divideFrameIntoMacroblocks(prev_frame, macroblock_size, width, height)
    curr_macroblocks_SAD_values = getSADErrorValue(curr_frame_macroblocks, prev_frame_macroblocks, k)
