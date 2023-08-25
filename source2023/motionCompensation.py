import numpy as np

macroblockSize = 64


def motionCompensationForEncoding(frames, motionVectors, width):
    """
        Motion compensation for encoding.
    """
    noOfCols = width // macroblockSize
    motionCompensatedFrames = [frames[0]]  # I frame

    for i in range(1, len(frames)):
        # Get the reference (previous) and target (current) frame
        referenceFrame = frames[i - 1]
        targetFrame = np.zeros_like(referenceFrame)
        idxOfVectorsForCurrFrame = i - 1
        noOfMacroblocks = len(motionVectors[idxOfVectorsForCurrFrame])

        for j in range(noOfMacroblocks):
            # Get the motion vector of the current macroblock
            motionVector = motionVectors[idxOfVectorsForCurrFrame][j][0]

            # Get the starting pixel of the current macroblock
            macroblockIdx = (j % noOfCols, j // noOfCols)  # (x, y)
            startingRefPixel = (macroblockIdx[0] * macroblockSize, macroblockIdx[1] * macroblockSize)  # (x, y)
            targetFrame = motionCompensationForEncodingOnSpecificFrame(motionVector, targetFrame, referenceFrame,
                                                                       startingRefPixel)

        motionCompensatedFrames.append(targetFrame)
    return motionCompensatedFrames


def motionCompensationForDecoding(iFrame, motionVectors):
    """
        Motion compensation for decoding.
    """
    motionCompensatedFrames = [iFrame]
    for i in range(len(motionVectors)):  # len(motionVectors) equals to the number of frames
        referenceFrame = motionCompensatedFrames[-1]  # Reference frame is the last frame in the list
        targetFrame = referenceFrame.copy()
        for j in range(len(motionVectors[i])):  # len(motionVectors[i]) equals to the number of macroblocks in the frame
            startingPixel = (j // macroblockSize, j % macroblockSize)  # (x, y)
            motionVector = motionVectors[i][j]  # Get the motion vector of the current macroblock, (x, y)
        motionCompensatedFrames.append(
            motionCompensationForSpecificFrame(motionVector, targetFrame, referenceFrame, startingPixel))
    return motionCompensatedFrames


def motionCompensationForEncodingOnSpecificFrame(motionVector, targetFrame, referenceFrame, startingRefPixel):
    """
        Perform motion compensation on a specific frame.
    """
    '''if motionVector[0] < 0:  # x < 0
        if motionVector[1] < 0:  # y < 0
            # Move left and up
            return performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

        if motionVector[1] > 0:  # y > 0
            # Move left and down
            targetFrame[
            startingRefPixel[1]:startingRefPixel[1] - macroblockSize,
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize
            ] = referenceFrame[
                startingRefPixel[1]:startingRefPixel[1] - macroblockSize,
                startingRefPixel[0]:startingRefPixel[0] + macroblockSize
                ]
        else:  # y == 0
            # Move left
            targetFrame[
            startingRefPixel[1],
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize
            ] = referenceFrame[
                startingRefPixel[1],
                startingRefPixel[0]:startingRefPixel[0] + macroblockSize
                ]
    elif motionVector[0] > 0:  # x > 0
        if motionVector[1] < 0:  # y < 0
            # Move right and up
            targetFrame[
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize,
            startingRefPixel[0]:startingRefPixel[0] - macroblockSize
            ] = referenceFrame[
                startingRefPixel[1]:startingRefPixel[1] + macroblockSize,
                startingRefPixel[0]:startingRefPixel[0] - macroblockSize
                ]
        if motionVector[1] > 0:  # y > 0
            # Move right and down
            targetFrame[
            startingRefPixel[1]:startingRefPixel[1] - macroblockSize,
            startingRefPixel[0]:startingRefPixel[0] - macroblockSize
            ] = referenceFrame[
                startingRefPixel[1]:startingRefPixel[1] - macroblockSize,
                startingRefPixel[0]:startingRefPixel[0] - macroblockSize
                ]
        else:  # y == 0
            # Move right
            targetFrame[
            startingRefPixel[1],
            startingRefPixel[0]:startingRefPixel[0] - macroblockSize
            ] = referenceFrame[
                startingRefPixel[1],
                startingRefPixel[0]:startingRefPixel[0] - macroblockSize
                ]
    else:  # x == 0
        if motionVector[1] < 0:  # y < 0
            # Move up
            targetFrame[
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize,
            startingRefPixel[0]
            ] = referenceFrame[
                startingRefPixel[1]:startingRefPixel[1] + macroblockSize,
                startingRefPixel[0]
                ]
        if motionVector[1] > 0:  # y > 0
            # Move down
            targetFrame[
            startingRefPixel[1]:startingRefPixel[1] - macroblockSize,
            startingRefPixel[0]
            ] = referenceFrame[
                startingRefPixel[1]:startingRefPixel[1] - macroblockSize,
                startingRefPixel[0]
                ]
        else:  # y == 0
            # No movement
            targetFrame = referenceFrame'''
    startingTargetPixel = (startingRefPixel[0] + motionVector[0], startingRefPixel[1] + motionVector[1])
    targetFrame[
    startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize,
    startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize
    ] = referenceFrame[
        startingRefPixel[1]:startingRefPixel[1] + macroblockSize,
        startingRefPixel[0]:startingRefPixel[0] + macroblockSize
        ]
    return targetFrame


def performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame):
    """
        Perform motion compensation.
    """
    startingTargetPixel = (startingRefPixel[0] + motionVector[0], startingRefPixel[1] + motionVector[1])

    if startingTargetPixel[0] < 0 and startingTargetPixel[1] < 0:
        # out of bounds (top left)
        return True

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and \
            startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (bottom right)
        return True

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and startingTargetPixel[1] < 0:
        # out of bounds (top right)
        return True

    if startingTargetPixel[0] < 0 and startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (bottom left)
        return True

    if startingTargetPixel[0] < 0:
        # out of bounds (left)
        return True

    if startingTargetPixel[1] < 0:
        # out of bounds (top)
        return True

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0]:
        # out of bounds (right)
        return True

    if startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (bottom)
        return True

    # if no out of bounds
    targetFrame[
    startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize,
    startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize
    ] = referenceFrame[
        startingRefPixel[1]:startingRefPixel[1] + macroblockSize,
        startingRefPixel[0]:startingRefPixel[0] + macroblockSize
        ]

    return targetFrame
