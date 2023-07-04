import cv2
import numpy as np


def hierarchicalSearch(frames, width, height):
    """
        Execute the hierarchical search algorithm.
    """
    motion_vectors = []
    for P in range(1, len(frames)):
        print("Hierarchical search for frame: ", P)
        prev_frame = frames[P - 1]
        curr_frame = frames[P]

        # Subsample previous and current frames in 3 levels
        curr_frame_levels = getFrameLevels(curr_frame, width, height)
        prev_frame_levels = getFrameLevels(prev_frame, width, height)

        # Execute the hierarchical search algorithm for each level
        levels = len(curr_frame_levels)
        MV_n_SAD = None
        temp_width = width
        temp_height = height
        for level in range(0, levels):
            macroblock_size = getMacroblockSize(level)
            temp_width, temp_height = getWidthAndHeight(level, temp_width, temp_height)
            MV_n_SAD = executeLevel(MV_n_SAD, curr_frame_levels[level], prev_frame_levels[level], macroblock_size,
                                    temp_width, temp_height)
        motion_vectors.append([(x * 4, y * 4) for (x, y), SAD in MV_n_SAD])  # multiply by 4 because we subsampled the
        # frames in 3 levels
    return motion_vectors


def getFrameLevels(frame, width, height):
    """
        Subsample the frame in 3 levels.
    """
    return [frame,
            cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR),
            cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)]


def getMacroblockSize(level):
    """
        Get the macroblock size for each level.
    """
    if level == 0:
        return 64
    elif level == 1:
        return 32
    return 16  # level 2


def getWidthAndHeight(level, width, height):
    """
        Get the width and height for each level.
    """
    if level == 0:
        return width, height
    return width // 2, height // 2  # level 1 and 2


def executeLevel(MV_n_SAD, curr_frame, prev_frame, macroblock_size, width, height):
    """
    Execute the hierarchical search algorithm for each level. Because search radius (k) is always smaller than the
    macroblock size, the search window is always to the nearest neighbors of the current macroblock. So our
    implementation practically doesn't need the variable k.
    """
    no_of_cols = width // macroblock_size  # number of macroblocks per row
    no_of_rows = height // macroblock_size  # number of macroblocks per column
    curr_frame_macroblocks = divideFrameIntoMacroblocks(curr_frame, macroblock_size, width, height, no_of_rows,
                                                        no_of_cols)
    prev_frame_macroblocks = divideFrameIntoMacroblocks(prev_frame, macroblock_size, width, height, no_of_rows,
                                                        no_of_cols)
    matched_macroblocks = getSADErrorValues(curr_frame_macroblocks, prev_frame_macroblocks, no_of_rows, no_of_cols)

    if MV_n_SAD is None:  # first level
        return calculateMotionVectors(matched_macroblocks, macroblock_size, no_of_cols)  # return the motion vectors and
    # SAD values

    # second and third level
    MV_n_SAD = [[(x // 2, y // 2), SAD] for (x, y), SAD in MV_n_SAD]  # divide the motion vectors by 2
    MV_n_SAD_new = calculateMotionVectors(matched_macroblocks, macroblock_size, no_of_cols)
    return compareMV_n_SAD(MV_n_SAD, MV_n_SAD_new)  # return the updated motion vectors and SAD values, if needed


def divideFrameIntoMacroblocks(frame, macroblock_size, width, height, no_of_rows, no_of_cols):
    """
        Divide the frame into macroblocks.
    """
    macroblocks = [[0 for _ in range(no_of_cols)] for _ in range(no_of_rows)]  # 2D array
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

    for i in range(no_of_rows):
        for j in range(no_of_cols):
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
        motion_vector = (curr_pixel[0] - prev_pixel[0], curr_pixel[1] - prev_pixel[1])
        SAD_value = matched_macroblocks[i][1]  # SAD value of the matched macroblock
        MV_n_SAD.append([motion_vector, SAD_value])  # Append the motion vector and the SAD value to the list
    return MV_n_SAD


def compareMV_n_SAD(MV_n_SAD, MV_n_SAD_new):
    """
        Compare and update the motion vectors and SAD values, if necessary.
    """
    for i in range(len(MV_n_SAD)):
        if MV_n_SAD[i][1] > MV_n_SAD_new[i][1]:
            MV_n_SAD[i] = MV_n_SAD_new[i]
    return MV_n_SAD
