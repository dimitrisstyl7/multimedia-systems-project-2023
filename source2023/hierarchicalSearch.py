import cv2
import numpy as np


def hierarchicalSearch(frames, width, height):
    """
        Execute the hierarchical search algorithm.
    """
    motion_vectors = []
    for P in range(1, len(frames)):
        prev_frame = frames[30]
        curr_frame = frames[31]

        # Subsample previous and current frames in 3 levels
        curr_frame_levels = getFrameLevels(curr_frame, width, height)
        prev_frame_levels = getFrameLevels(prev_frame, width, height)

        # Execute the hierarchical search algorithm for each level
        levels = len(curr_frame_levels)
        for level in range(0, levels):
            macroblock_size = getMacroblockSize(level)
            k = getSearchRadius(level)  # search radius (in pixels)
            width, height = getWidthAndHeight(level, width, height)
            executeLevel(curr_frame_levels[level], prev_frame_levels[level], macroblock_size, width, height)


def getFrameLevels(frame, width, height):
    return [frame,
            cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR),
            cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)]


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


def executeLevel(curr_frame, prev_frame, macroblock_size, width, height):
    no_of_cols = width // macroblock_size  # number of macroblocks per row
    no_of_rows = height // macroblock_size  # number of macroblocks per column
    curr_frame_macroblocks = divideFrameIntoMacroblocks(curr_frame, macroblock_size, width, height, no_of_rows,
                                                        no_of_cols)
    prev_frame_macroblocks = divideFrameIntoMacroblocks(prev_frame, macroblock_size, width, height, no_of_rows,
                                                        no_of_cols)
    matched_macroblocks = getSADErrorValues(curr_frame_macroblocks, prev_frame_macroblocks, no_of_rows, no_of_cols)
    MV_n_SAD = calculateMotionVectors(matched_macroblocks, macroblock_size, no_of_cols)


def divideFrameIntoMacroblocks(frame, macroblock_size, width, height, no_of_rows, no_of_cols):
    """
        Divide the frame into macroblocks.
    """
    macroblocks = [[0 for i in range(no_of_cols)] for j in range(no_of_rows)]  # 2D array
    i, j = 0, 0

    for row in range(0, height, macroblock_size):
        for col in range(0, width, macroblock_size):
            macroblocks[i][j] = frame[row: row + macroblock_size, col: col + macroblock_size]
            j += 1
        i += 1
        j = 0
    return np.array(macroblocks)


def getSADErrorValues(curr_frame_macroblocks, prev_frame_macroblocks, no_of_rows, no_of_cols):
    """
        Sum of Absolute Differences (SAD) error function.
        Because k is always smaller than the macroblock size, the search window is always to the nearest neighbors of
        the current macroblock.
    """
    matched_macroblocks = []  # Matched macroblocks, contains lists in the form: [(i, j), SAD_value] where (i, j)
    # is the coordinate of the matched macroblock in the previous frame

    for i in range(len(curr_frame_macroblocks)):
        for j in range(len(curr_frame_macroblocks[i])):
            neigh_prev_macroblocks_coord = [(i, j)]  # Contains the coordinates of the previous macroblocks that are
            # neighbors of the current macroblock
            neigh_prev_SAD_values = \
                [calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i][j])]  # Contains the SAD
            # values of the previous macroblocks that are neighbors of the current macroblock

            if j > 0:  # left previous macroblock
                neigh_prev_macroblocks_coord.append((i, j - 1))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i][j - 1]))

            if j < no_of_cols - 1:  # right previous macroblock
                neigh_prev_macroblocks_coord.append((i, j + 1))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i][j + 1]))

            if i > 0:  # top previous macroblock
                neigh_prev_macroblocks_coord.append((i - 1, j))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i - 1][j]))

            if i < no_of_rows - 1:  # bottom previous macroblock
                neigh_prev_macroblocks_coord.append((i + 1, j))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i + 1][j]))

            if i > 0 and j > 0:  # top-left previous macroblock
                neigh_prev_macroblocks_coord.append((i - 1, j - 1))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i - 1][j - 1]))

            if i > 0 and j < no_of_cols - 1:  # top-right previous macroblock
                neigh_prev_macroblocks_coord.append((i - 1, j + 1))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i - 1][j + 1]))

            if i < no_of_rows - 1 and j > 0:  # bottom-left previous macroblock
                neigh_prev_macroblocks_coord.append((i + 1, j - 1))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i + 1][j - 1]))

            if i < no_of_rows - 1 and j < no_of_cols - 1:  # bottom-right previous macroblock
                neigh_prev_macroblocks_coord.append((i + 1, j + 1))
                neigh_prev_SAD_values.append(
                    calculateSADValue(curr_frame_macroblocks[i][j], prev_frame_macroblocks[i + 1][j + 1]))

            min_SAD = min(neigh_prev_SAD_values)  # Minimum SAD value
            prev_macroblock_coord = neigh_prev_macroblocks_coord[neigh_prev_SAD_values.index(min_SAD)]  # Previous
            # macroblock coordinates with the minimum SAD value
            matched_macroblocks.append([prev_macroblock_coord, min_SAD])  # Append the matched macroblock coordinates
            # and the minimum SAD value to the list
    return matched_macroblocks


def calculateSADValue(curr_macroblock, prev_macroblock):
    """
        Calculate the SAD value between the current macroblock and the previous macroblock.
    """
    SAD = 0
    for i in range(curr_macroblock.shape[0]):
        for j in range(curr_macroblock.shape[1]):
            SAD += abs(int(curr_macroblock[i, j]) - int(prev_macroblock[i, j]))
    return SAD


def calculateMotionVectors(matched_macroblocks, macroblock_size, no_of_cols):
    """
        Calculate the motion vectors for each matched macroblock.
    """
    MV_n_SAD = []  # Contains lists in the form: [motion_vector, SAD_value]
    dx, dy = 0, 0
    for i in range(len(matched_macroblocks)):
        # Find the coordinates of the current macroblock
        curr_macroblock_row = i // no_of_cols
        curr_macroblock_col = i % no_of_cols

        # Find the real pixel coordinates of the current macroblock (top-left corner), form: (x, y)
        curr_pixel = (curr_macroblock_col * macroblock_size, curr_macroblock_row * macroblock_size)

        # Find the coordinates of the previous macroblock
        prev_macroblock_row, prev_macroblock_col = matched_macroblocks[i][0]

        # Find the real pixel coordinates of the previous macroblock (top-left corner), form: (x, y)
        prev_pixel = (prev_macroblock_col * macroblock_size, prev_macroblock_row * macroblock_size)

        # Calculate the motion vector, form: (dx, dy)
        motion_vector = (dx + curr_pixel[0] - prev_pixel[0], dy + curr_pixel[1] - prev_pixel[1])
        SAD_value = matched_macroblocks[i][1]  # SAD value of the matched macroblock
        MV_n_SAD.append([motion_vector, SAD_value])  # Append the motion vector and the SAD value to the list
    return MV_n_SAD
