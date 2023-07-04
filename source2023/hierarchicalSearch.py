import cv2
import numpy as np


def hierarchicalSearch(frames, width, height):
    """
        Execute the hierarchical search algorithm.
    """
    motionVectors = []
    for P in range(1, len(frames)):
        print("Hierarchical search for frame: ", P)
        prevFrame = frames[P - 1]
        currFrame = frames[P]

        # Subsample previous and current frames in 3 levels
        currFrameLevels = getFrameLevels(currFrame, width, height)
        prevFrameLevels = getFrameLevels(prevFrame, width, height)

        # Execute the hierarchical search algorithm for each level
        levels = len(currFrameLevels)
        MVnSAD = None
        tempWidth = width
        tempHeight = height
        for level in range(0, levels):
            macroblockSize = getMacroblockSize(level)
            tempWidth, tempHeight = getWidthAndHeight(level, tempWidth, tempHeight)
            MVnSAD = executeLevel(MVnSAD, currFrameLevels[level], prevFrameLevels[level], macroblockSize,
                                  tempWidth, tempHeight)
        motionVectors.append([(x * 4, y * 4) for (x, y), SAD in MVnSAD])  # multiply by 4 because we subsampled the
        # frames in 3 levels
    return motionVectors


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


def executeLevel(MVnSAD, currFrame, prevFrame, macroblockSize, width, height):
    """
        Execute the hierarchical search algorithm for each level. Because search radius (k) is always smaller than the
        macroblock size, the search window is always to the nearest neighbors of the current macroblock. So our
        implementation practically doesn't need the variable k.
    """
    noOfCols = width // macroblockSize  # number of macroblocks per row
    noOfRows = height // macroblockSize  # number of macroblocks per column
    currFrameMacroblocks = divideFrameIntoMacroblocks(currFrame, macroblockSize, width, height, noOfRows,
                                                      noOfCols)
    prevFrameMacroblocks = divideFrameIntoMacroblocks(prevFrame, macroblockSize, width, height, noOfRows,
                                                      noOfCols)
    matchedMacroblocks = getSADErrorValues(currFrameMacroblocks, prevFrameMacroblocks, noOfRows, noOfCols)

    if MVnSAD is None:  # first level
        return calculateMotionVectors(matchedMacroblocks, macroblockSize, noOfCols)  # return the motion vectors and
    # SAD values

    # second and third level
    MVnSAD = [[(x // 2, y // 2), SAD] for (x, y), SAD in MVnSAD]  # divide the motion vectors by 2
    MVnSAD_new = calculateMotionVectors(matchedMacroblocks, macroblockSize, noOfCols)
    return compareMVnSAD(MVnSAD, MVnSAD_new)  # return the updated motion vectors and SAD values, if needed


def divideFrameIntoMacroblocks(frame, macroblockSize, width, height, noOfRows, noOfCols):
    """
        Divide the frame into macroblocks.
    """
    macroblocks = [[0 for _ in range(noOfCols)] for _ in range(noOfRows)]  # 2D array
    i, j = 0, 0

    for row in range(0, height, macroblockSize):
        for col in range(0, width, macroblockSize):
            macroblocks[i][j] = frame[row: row + macroblockSize, col: col + macroblockSize]
            j += 1
        i += 1
        j = 0
    return np.array(macroblocks)


def getSADErrorValues(currFrameMacroblocks, prevFrameMacroblocks, noOfRows, noOfCols):
    """
        Sum of Absolute Differences (SAD) error function.
        Because k is always smaller than the macroblock size, the search window is always to the nearest neighbors of
        the current macroblock.
    """
    matchedMacroblocks = []  # Matched macroblocks, contains lists in the form: [(i, j), SAD_value] where (i, j)
    # is the coordinate of the matched macroblock in the previous frame

    for i in range(noOfRows):
        for j in range(noOfCols):
            neighPrevMacroblocksCoord = [(i, j)]  # Contains the coordinates of the previous macroblocks that are
            # neighbors of the current macroblock
            neighPrevSAD_values = \
                [calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i][j])]  # Contains the SAD
            # values of the previous macroblocks that are neighbors of the current macroblock

            if j > 0:  # left previous macroblock
                neighPrevMacroblocksCoord.append((i, j - 1))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i][j - 1]))

            if j < noOfCols - 1:  # right previous macroblock
                neighPrevMacroblocksCoord.append((i, j + 1))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i][j + 1]))

            if i > 0:  # top previous macroblock
                neighPrevMacroblocksCoord.append((i - 1, j))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i - 1][j]))

            if i < noOfRows - 1:  # bottom previous macroblock
                neighPrevMacroblocksCoord.append((i + 1, j))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i + 1][j]))

            if i > 0 and j > 0:  # top-left previous macroblock
                neighPrevMacroblocksCoord.append((i - 1, j - 1))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i - 1][j - 1]))

            if i > 0 and j < noOfCols - 1:  # top-right previous macroblock
                neighPrevMacroblocksCoord.append((i - 1, j + 1))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i - 1][j + 1]))

            if i < noOfRows - 1 and j > 0:  # bottom-left previous macroblock
                neighPrevMacroblocksCoord.append((i + 1, j - 1))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i + 1][j - 1]))

            if i < noOfRows - 1 and j < noOfCols - 1:  # bottom-right previous macroblock
                neighPrevMacroblocksCoord.append((i + 1, j + 1))
                neighPrevSAD_values.append(
                    calculateSADValue(currFrameMacroblocks[i][j], prevFrameMacroblocks[i + 1][j + 1]))

            minSAD = min(neighPrevSAD_values)  # Minimum SAD value
            prevMacroblockCoord = neighPrevMacroblocksCoord[neighPrevSAD_values.index(minSAD)]  # Previous
            # macroblock coordinates with the minimum SAD value
            matchedMacroblocks.append([prevMacroblockCoord, minSAD])  # Append the matched macroblock coordinates
            # and the minimum SAD value to the list
    return matchedMacroblocks


def calculateSADValue(currMacroblock, prevMacroblock):
    """
        Calculate the SAD value between the current macroblock and the previous macroblock.
    """
    SAD = 0
    for i in range(currMacroblock.shape[0]):
        for j in range(currMacroblock.shape[1]):
            SAD += abs(int(currMacroblock[i, j]) - int(prevMacroblock[i, j]))
    return SAD


def calculateMotionVectors(matchedMacroblocks, macroblockSize, noOfCols):
    """
        Calculate the motion vectors for each matched macroblock.
    """
    MVnSAD = []  # Contains lists in the form: [motionVector, SAD_value]
    for i in range(len(matchedMacroblocks)):
        # Find the coordinates of the current macroblock
        currMacroblockRow = i // noOfCols
        currMacroblockCol = i % noOfCols

        # Find the real pixel coordinates of the current macroblock (top-left corner), form: (x, y)
        currPixel = (currMacroblockCol * macroblockSize, currMacroblockRow * macroblockSize)

        # Find the coordinates of the previous macroblock
        prevMacroblockRow, prevMacroblockCol = matchedMacroblocks[i][0]

        # Find the real pixel coordinates of the previous macroblock (top-left corner), form: (x, y)
        prevPixel = (prevMacroblockCol * macroblockSize, prevMacroblockRow * macroblockSize)

        # Calculate the motion vector, form: (dx, dy)
        motionVector = (currPixel[0] - prevPixel[0], currPixel[1] - prevPixel[1])
        SAD_value = matchedMacroblocks[i][1]  # SAD value of the matched macroblock
        MVnSAD.append([motionVector, SAD_value])  # Append the motion vector and the SAD value to the list
    return MVnSAD


def compareMVnSAD(MVnSAD, MVnSAD_new):
    """
        Compare and update the motion vectors and SAD values, if necessary.
    """
    for i in range(len(MVnSAD)):
        if MVnSAD[i][1] > MVnSAD_new[i][1]:
            MVnSAD[i] = MVnSAD_new[i]
    return MVnSAD
