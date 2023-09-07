import numpy as np

from progressBar import *

macroblockSize = 64


def motionCompensation(frames, motionVectors, width, height):
    noOfCols = width // macroblockSize
    motionCompensatedFrames = [frames[0]]  # I frame
    backgroundFrame = frames[0]  # I frame

    progressBar(0, len(frames), 'Creating Motion Compensation Frames: ', 'Motion Compensation Frames Created!')
    for i in range(1, len(frames)):
        # Get the reference (previous) and target (current) frame
        referenceFrame = frames[i - 1]
        targetFrame = frames[i]
        # targetFrame = np.zeros_like(referenceFrame)
        idxOfVectorsForCurrFrame = i - 1
        noOfMacroblocks = len(motionVectors[idxOfVectorsForCurrFrame])

        for j in range(noOfMacroblocks):
            # Get the motion vector of the current macroblock
            motionVector = motionVectors[idxOfVectorsForCurrFrame][j]

            # Get the starting pixel of the current macroblock
            macroblockIdx = (j // noOfCols, j % noOfCols)  # (y, x)
            startingRefPixel = (macroblockIdx[0] * macroblockSize, macroblockIdx[1] * macroblockSize)  # (y, x)
            targetFrame = motionCompensationOnSpecificFrame(motionVector, targetFrame, backgroundFrame,
                                                            startingRefPixel, width, height)
        motionCompensatedFrames.append(targetFrame)
        progressBar(i + 1, len(frames), 'Creating Motion Compensation Frames: ', 'Motion Compensation Frames Created!')
    return motionCompensatedFrames


def motionCompensationOnSpecificFrame(motionVector, targetFrame, backgroundFrame,
                                      startingRefPixel, width, height):
    """
        Motion compensation on a specific frame.
    """
    if motionVector[0] == 0 and motionVector[1] == 0:
        # No movement
        return targetFrame
    return performMotionCompensation(startingRefPixel, motionVector, targetFrame, backgroundFrame,
                                     width, height)


def performMotionCompensation(startingRefPixel, motionVector, targetFrame, backgroundFrame,
                              width, height):
    """
        Perform motion compensation.
    """
    startingTargetPixel = (startingRefPixel[0] + motionVector[0], startingRefPixel[1] + motionVector[1])

    if startingTargetPixel[0] < 0 and startingTargetPixel[1] < 0:
        # out of bounds (top left)
        dy = startingTargetPixel[0] + macroblockSize
        dx = startingTargetPixel[1] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and \
            startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (bottom right)
        dy, dx = height - startingTargetPixel[0], width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]
        return targetFrame

    if startingTargetPixel[0] < 0 and startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (top right)
        dy = startingTargetPixel[0] + macroblockSize
        dx = width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]
        return targetFrame

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and startingTargetPixel[1] < 0:
        # out of bounds (bottom left)
        dy = height - startingTargetPixel[0]
        dx = startingTargetPixel[1] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    if startingTargetPixel[1] < 0:
        # out of bounds (left)
        dx = startingTargetPixel[1] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    if startingTargetPixel[0] < 0:
        # out of bounds (top)
        dy = startingTargetPixel[0] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    if startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (right)
        dx = width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]
        return targetFrame

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0]:
        # out of bounds (bottom)
        dy = height - startingTargetPixel[0]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = backgroundFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    # if no out of bounds
    targetFrame[
    startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
    startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
    ] = backgroundFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ]
    return targetFrame
