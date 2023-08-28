import cv2
import numpy as np

macroblockSize = 64
radius = 32  # Search radius
numLevels = 3  # Pyramid Levels


def hierarchicalSearch(referenceFrame, targetFrame, width, height):
    """
            Execute the hierarchical search algorithm.
    """
    motionVectors = []

    for i in range(1, len(originalFrames)):
        print(f'Frame {i} of {len(originalFrames) - 1}')
        referenceFrame = originalFrames[i - 1]
        targetFrame = originalFrames[i]

        # Subsample previous and current frames in 3 levels
        referenceFramePyramid, targetFramePyramid = createPyramidLevels(referenceFrame, targetFrame)

        # Execute the hierarchical search algorithm for each level
        levels = len(referenceFramePyramid)
        MVnSAD = None
        for level in range(levels - 1, -1, -1):
            MVnSAD = executeLevel(referenceFramePyramid[level], targetFramePyramid[level], level, MVnSAD)
        motionVectors.append([value[0] for value in MVnSAD])
    return motionVectors


def createPyramidLevels(referenceFrame, targetFrame):
    """
        Create the Gaussian pyramid for the two frames.
    """
    referenceFramePyramid = [referenceFrame]
    targetFramePyramid = [targetFrame]

    for level in range(1, numLevels):
        referenceFrame = cv2.pyrDown(referenceFrame)
        targetFrame = cv2.pyrDown(targetFrame)
        referenceFramePyramid.append(referenceFrame)
        targetFramePyramid.append(targetFrame)

    return referenceFramePyramid, targetFramePyramid


def executeLevel(referenceFrame, targetFrame, level, MVnSAD):
    """
        Execute the hierarchical search algorithm for each level.
    """
    width = referenceFrame.shape[1]
    height = referenceFrame.shape[0]
    k = radius // (2 ** level)
    macroblockLevelSize = macroblockSize // (2 ** level)
    noOfCols = width // macroblockLevelSize  # number of macroblocks per row
    noOfRows = height // macroblockLevelSize  # number of macroblocks per column
    referenceFrameInMacroblocks = divideFrameIntoMacroblocks(referenceFrame, macroblockLevelSize, width, height,
                                                             noOfRows, noOfCols)
    targetFrameInMacroblocks = divideFrameIntoMacroblocks(targetFrame, macroblockLevelSize, width, height, noOfRows,
                                                          noOfCols)

    if level == 2:  # level 3 (highest level - executing full search algorithm)
        return getMVnSADErrorValuesForHighestLevel(referenceFrame, targetFrameInMacroblocks, macroblockLevelSize,
                                                   noOfRows, noOfCols, k)
    else:  # levels 1-2 (executing block-matching algorithm)
        MVnSAD_old = [[(y * 2, x * 2), SAD] for (y, x), SAD in MVnSAD]
        matchedMacroblocks = getSADErrorValues(targetFrameInMacroblocks, referenceFrameInMacroblocks, noOfRows,
                                               noOfCols)
        MVnSAD_new = calculateMotionVectors(matchedMacroblocks, macroblockLevelSize, noOfCols)
        return compareMVnSAD(MVnSAD_old, MVnSAD_new)  # return the updated motion vectors and SAD values, if needed


def divideFrameIntoMacroblocks(frame, macroblockLevelSize, width, height, noOfRows, noOfCols):
    """
        Divide the frame into macroblocks.
    """
    macroblocks = [[0 for _ in range(noOfCols)] for _ in range(noOfRows)]  # 2D array
    i, j = 0, 0

    for row in range(0, height, macroblockLevelSize):
        for col in range(0, width, macroblockLevelSize):
            macroblocks[i][j] = frame[row: row + macroblockLevelSize, col: col + macroblockLevelSize]
            j += 1
        i += 1
        j = 0
    return np.array(macroblocks)


def getMVnSADErrorValuesForHighestLevel(referenceFrame, targetFrameInMacroblocks, macroblockLevelSize, noOfRows,
                                        noOfCols, k):
    MVnSAD = []  # MVnSAD list form:
    # [ [MV_for_i_macroblock, SAD_for_i_macroblock], [MV_for_i+1_macroblock, SAD_for_i+1_macroblock], ... ]
    for i in range(noOfRows):
        for j in range(noOfCols):
            targetMacroblock = targetFrameInMacroblocks[i][j]

            # Find the coordinates of the pixel (top-left corner) on the target macroblock, which is equivalent of
            # the pixel on reference frame. targetPixel form: (y, x)
            targetPixel = (i * macroblockLevelSize, j * macroblockLevelSize)

            # Find the search area on the reference frame that is inside the given radius (k).
            startingPixel, endingPixel = findSearchArea(targetPixel, k, referenceFrame.shape[1],
                                                        referenceFrame.shape[0])
            MVnSAD.append(
                executeFullSearch(referenceFrame, targetMacroblock, targetPixel, macroblockLevelSize, startingPixel,
                                  endingPixel)
            )
    return MVnSAD


def findSearchArea(targetPixel, k, width, height):
    """
        Find the search area on the reference frame that is inside the given radius (k).
    """
    startingPixel = (targetPixel[0] - k, targetPixel[1] - k)  # Starting pixel of the search area (top-left corner),
    # Pixel form: (y, x)
    if startingPixel[0] < 0:  # If the starting pixel is out of the y-axis, set y = 0
        startingPixel = (0, startingPixel[1])
    if startingPixel[1] < 0:  # If the starting pixel is out of the x-axis, set x = 0
        startingPixel = (startingPixel[0], 0)

    # Ending pixel of the search area (bottom-right),
    # Pixel form: (y, x)
    endingPixel = (targetPixel[0] + k, targetPixel[1] + k)  # Ending pixel of the search area.
    # == (targetPixel[0] + macroblockSize + k) - macroblockSize, (targetPixel[1] + macroblockSize + k) - macroblockSize
    if endingPixel[0] > height:  # If the ending pixel is out of the y-axis, set y = height
        endingPixel = (height, endingPixel[1])
    if endingPixel[1] > width:  # If the ending pixel is out of the x-axis, set x = width
        endingPixel = (endingPixel[0], width)

    return startingPixel, endingPixel


def executeFullSearch(referenceFrame, targetMacroblock, targetPixel, macroblockLevelSize, startingPixel, endingPixel):
    """
        Execute full search algorithm for the target macroblock.
    """
    SAD_values = []  # SAD_values list form: [SAD_value_i, SAD_value_i+1, ...]
    pixels = []  # pixels list form: [(pixel_i_y, pixel_i_x), (pixel_i+1_y, pixel_i+1_x), ...]

    for row in range(startingPixel[0], endingPixel[0] + 1):
        for col in range(startingPixel[1], endingPixel[1] + 1):
            referenceMacroblock = constructTempReferenceMacroblock(referenceFrame, row, col, macroblockLevelSize)
            SAD_values.append(calculateSADValue(referenceMacroblock, targetMacroblock))
            pixels.append((row, col))

    minSAD = min(SAD_values)  # Minimum SAD value
    referencePixel = pixels[SAD_values.index(minSAD)]

    # Calculate motion vector, form: (dy, dx)
    motionVector = (referencePixel[0] - targetPixel[0], referencePixel[1] - targetPixel[1])
    return [motionVector, minSAD]


def constructTempReferenceMacroblock(referenceFrame, row, col, macroblockLevelSize):
    return referenceFrame[row: row + macroblockLevelSize, col: col + macroblockLevelSize]


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
    matchedMacroblocks = []  # matchedMacroblocks list form: [(i, j), SAD_value] where (i, j) is the coordinate
    # of the matched macroblock in the reference frame

    for i in range(noOfRows):
        for j in range(noOfCols):
            neighRefMacroblocksCoord = [(i, j)]  # Contains the coordinates of the reference macroblocks that are
            # neighbors of the target macroblock
            neighRefSAD_values = \
                [calculateSADValue(targetFrameInMacroblocks[i][j],
                                   referenceFrameInMacroblocks[i][j])]  # Contains the SAD
            # values of the reference macroblocks that are neighbors of the target macroblock

            if j > 0:  # left reference macroblock
                neighRefMacroblocksCoord.append((i, j - 1))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i][j - 1]))

            if j < noOfCols - 1:  # right reference macroblock
                neighRefMacroblocksCoord.append((i, j + 1))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i][j + 1]))

            if i > 0:  # top reference macroblock
                neighRefMacroblocksCoord.append((i - 1, j))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i - 1][j]))

            if i < noOfRows - 1:  # bottom reference macroblock
                neighRefMacroblocksCoord.append((i + 1, j))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i + 1][j]))

            if i > 0 and j > 0:  # top-left reference macroblock
                neighRefMacroblocksCoord.append((i - 1, j - 1))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i - 1][j - 1]))

            if i > 0 and j < noOfCols - 1:  # top-right reference macroblock
                neighRefMacroblocksCoord.append((i - 1, j + 1))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i - 1][j + 1]))

            if i < noOfRows - 1 and j > 0:  # bottom-left reference macroblock
                neighRefMacroblocksCoord.append((i + 1, j - 1))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i + 1][j - 1]))

            if i < noOfRows - 1 and j < noOfCols - 1:  # bottom-right reference macroblock
                neighRefMacroblocksCoord.append((i + 1, j + 1))
                neighRefSAD_values.append(
                    calculateSADValue(targetFrameInMacroblocks[i][j], referenceFrameInMacroblocks[i + 1][j + 1]))

            minSAD = min(neighRefSAD_values)  # Minimum SAD value
            refMacroblockCoord = neighRefMacroblocksCoord[neighRefSAD_values.index(minSAD)]  # Reference
            # macroblock coordinates with the minimum SAD value
            matchedMacroblocks.append([refMacroblockCoord, minSAD])  # Append the matched macroblock coordinates
            # and the minimum SAD value to the list
    return matchedMacroblocks


def calculateMotionVectors(matchedMacroblocks, macroblockLevelSize, noOfCols):
    """
        Calculate the motion vectors for each matched macroblock.
    """
    MVnSAD = []  # MVnSAD list form: [ [MV_for_i_matchedMacroblock, SAD_for_i_matchedMacroblock],
    # [MV_for_i+1_matchedMacroblock, SAD_for_i+1_matchedMacroblock], ... ]

    for i in range(len(matchedMacroblocks)):
        # Find the coordinates of the target macroblock in a frame (noOfCol x noOfRows)
        targetMacroblockRow = i // noOfCols
        targetMacroblockCol = i % noOfCols

        # Find the coordinates of the pixel (top-left corner) on the target macroblock, which is equivalent of the pixel
        # on reference frame. Pixel form: (y, x)
        targetPixel = (targetMacroblockRow * macroblockLevelSize, targetMacroblockCol * macroblockLevelSize)

        # Find the coordinates of the reference macroblock
        refMacroblockRow, refMacroblockCol = matchedMacroblocks[i][0]

        # Find the coordinates of the pixel (top-left corner) on the reference macroblock, which is equivalent of the
        # pixel on reference frame. refPixel form: (y, x)
        referencePixel = (refMacroblockRow * macroblockLevelSize, refMacroblockCol * macroblockLevelSize)

        # Calculate the motion vector, motionVector form: (dy, dx)
        motionVector = (referencePixel[0] - targetPixel[0], referencePixel[1] - targetPixel[1])
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
