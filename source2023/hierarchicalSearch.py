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

def getRadiusK(level):
    """
        Get the search radius for each level.
    """
    if level == 0:
        return 32
    elif level == 1:
        return 16
    return 8  # level 2


def getWidthAndHeight(level, width, height):
    """
        Get the width and height for each level.
    """
    if level == 0:
        return width, height
    elif level == 1:
        return width // 2, height // 2
    return width // 4, height // 4  # level 2


def executeLevel(referenceFrame, targetFrame, width, height, macroblockSize, level, MVnSAD, k):
    noOfCols = width // macroblockSize  # number of macroblocks per row
    noOfRows = height // macroblockSize  # number of macroblocks per column
    referenceFrameInMacroblocks = divideFrameIntoMacroblocks(referenceFrame, macroblockSize, width, height, noOfRows,
                                                             noOfCols)
    targetFrameInMacroblocks = divideFrameIntoMacroblocks(targetFrame, macroblockSize, width, height, noOfRows,
                                                          noOfCols)

    if level == 2:  # level 3 (highest level - executing full search algorithm)
        return getMVnSADErrorValuesForHighestLevel(referenceFrame, targetFrameInMacroblocks, macroblockSize, noOfCols, k)
    else:  # levels 1-2 (executing block-matching algorithm)
        MVnSAD_old = [[(x * 2, y * 2) for (x, y), SAD in MVnSAD]]
        matchedMacroblocks = getSADErrorValues(targetFrameInMacroblocks, referenceFrameInMacroblocks, noOfRows, noOfCols)
        MVnSAD_new = calculateMotionVectors(matchedMacroblocks, macroblockSize, noOfCols)
        return compareMVnSAD(MVnSAD_old, MVnSAD_new)  # return the updated motion vectors and SAD values, if needed


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


def getMVnSADErrorValuesForHighestLevel(referenceFrame, targetFrameInMacroblocks, macroblockSize, noOfCols, k):
    MVnSAD = []  # MVnSAD list form:
    # [ [MV_for_i_macroblock, SAD_for_i_macroblock], [MV_for_i+1_macroblock, SAD_for_i+1_macroblock], ... ]
    for i in range(len(targetFrameInMacroblocks)):
        targetMacroblock = targetFrameInMacroblocks[i]

        # Find the coordinates of the target macroblock on a frame (noOfCol x noOfRows)
        targetMacroblockRow = i // noOfCols  # macroblock
        targetMacroblockCol = i % noOfCols  # macroblock

        # Find the coordinates of the pixel (top-left corner) on the target macroblock, which is equivalent of the pixel
        # on reference frame. targetPixel form: (x, y)
        targetPixel = (targetMacroblockCol * macroblockSize, targetMacroblockRow * macroblockSize)

        # Find the search area on the reference frame that is inside the given radius (k).
        startingPixel, searchAreaWidth, searchAreaHeight = \
            findSearchArea(targetPixel, macroblockSize, k, referenceFrame.shape[1], referenceFrame.shape[0])
        MVnSAD.append(
            executeFullSearch(referenceFrame, targetMacroblock, targetPixel, macroblockSize, startingPixel,
                              searchAreaWidth, searchAreaHeight)
        )
    return MVnSAD


def findSearchArea(targetPixel, macroblockSize, k, width, height):
    """
        Find the search area on the reference frame that is inside the given radius (k).
    """
    startingPixel = (targetPixel[0] - k, targetPixel[1] - k)  # Starting pixel of the search area (top-left corner),
    # Pixel form: (x, y)
    if startingPixel[0] < 0:  # If the starting pixel is out of the x-axis, set x = 0
        startingPixel = (0, startingPixel[1])
    if startingPixel[1] < 0:  # If the starting pixel is out of the y-axis, set y = 0
        startingPixel = (startingPixel[0], 0)

    endingPixel = (targetPixel[0] + 2 * k, targetPixel[1] + 2 * k)  # Ending pixel of the search area (bottom-right),
    # Pixel form: (x, y)
    if endingPixel[0] > width:  # If the ending pixel is out of the x-axis, set x = width
        endingPixel = (width, endingPixel[1])
    if endingPixel[1] > height:  # If the ending pixel is out of the y-axis, set y = height
        endingPixel = (endingPixel[0], height)

    # Calculate the search area width and height
    searchAreaWidth = endingPixel[0] - startingPixel[0] - macroblockSize
    searchAreaHeight = endingPixel[1] - startingPixel[1] - macroblockSize
    return startingPixel, searchAreaWidth, searchAreaHeight


def executeFullSearch(referenceFrame, targetMacroblock, targetPixel, macroblockSize, startingPixel, searchAreaWidth,
                      searchAreaHeight):
    """
        Execute full search algorithm for the target macroblock.
    """
    SAD_values = []  # SAD_values list form: [SAD_value_i, SAD_value_i+1, ...]
    pixels = []  # pixels list form: [(pixel_i_X, pixel_i_Y), (pixel_i+1_X, pixel_i+1_Y), ...]
    for row in range(startingPixel[1], searchAreaHeight):
        for col in range(startingPixel[0], searchAreaWidth):
            referenceMacroblock = constructTempReferenceMacroblock(referenceFrame, row, col, macroblockSize)
            SAD_values.append(calculateSADValue(referenceMacroblock, targetMacroblock))
            pixels.append((row, col))

    minSAD = min(SAD_values)  # Minimum SAD value
    referenceFramePixel = pixels[SAD_values.index(minSAD)]

    # Calculate motion vector
    motionVector = (targetPixel[0] - referenceFramePixel[0], targetPixel[1] - referenceFramePixel[1])
    return [motionVector, minSAD]


def constructTempReferenceMacroblock(referenceFrame, row, col, macroblockSize):
    return referenceFrame[row: row + macroblockSize, col: col + macroblockSize]


def calculateSADValue(referenceMacroblock, targetMacroblock):
    """
        Calculate the SAD value between the reference macroblock and the target macroblock.
    """
    SAD = 0
    for i in range(referenceMacroblock.shape[0]):
        for j in range(referenceMacroblock.shape[1]):
            SAD += abs(int(targetMacroblock[i, j]) - int(referenceMacroblock[i, j]))
    return SAD


def getSADErrorValues(targetFrameInMacroblocks, referenceFrameInMacroblocks, noOfRows, noOfCols):
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

        # Calculate the motion vector, motionVector form: (dx, dy)
        motionVector = (targetPixel[0] - refPixel[0], targetPixel[1] - refPixel[1])
        SAD_value = matchedMacroblocks[i][1]  # SAD value of the matched macroblocks
        MVnSAD.append([motionVector, SAD_value])  # Append the motion vector and the SAD value in MVnSAD list
    return MVnSAD


def compareMVnSAD(MVnSAD, MVnSAD_new):
    """
        Compare and update the motion vectors and SAD values, if necessary.
    """
    for i in range(len(MVnSAD)):
        if MVnSAD[i][1] > MVnSAD_new[i][1]:
            MVnSAD[i] = MVnSAD_new[i]
    return MVnSAD
