import numpy as np

macroblockSize = 64


def motionCompensationForEncoding(frames, motionVectors, width, height):
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
            motionVector = motionVectors[idxOfVectorsForCurrFrame][j]

            # Get the starting pixel of the current macroblock
            macroblockIdx = (j // noOfCols, j % noOfCols)  # (y, x)
            startingRefPixel = (macroblockIdx[0] * macroblockSize, macroblockIdx[1] * macroblockSize)  # (y, x)
            targetFrame = motionCompensationForEncodingOnSpecificFrame(motionVector, targetFrame, referenceFrame,
                                                                       startingRefPixel, width, height)
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
            startingPixel = (j % macroblockSize, j // macroblockSize)  # (y, x)
            motionVector = motionVectors[i][j]  # Get the motion vector of the current macroblock, (y, x)
        motionCompensatedFrames.append(
            motionCompensationForSpecificFrame(motionVector, targetFrame, referenceFrame, startingPixel))
    return motionCompensatedFrames


def motionCompensationForEncodingOnSpecificFrame(motionVector, targetFrame, referenceFrame, startingRefPixel, width,
                                                 height):
    """
        Perform motion compensation on a specific frame.
    """

    if motionVector[0] == 0 and motionVector[1] == 0:
        # No movement
        return referenceFrame
    return performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame, width, height)


def motionCompensationForDecodingOnSpecificFrame(motionVector, targetFrame, referenceFrame, startingRefPixel):
    """
        Perform motion compensation on a specific frame.
    """
    pass


def performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame, width, height):
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
        ] = referenceFrame[
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
        ] = referenceFrame[
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
        ] = referenceFrame[
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
        ] = referenceFrame[
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
        ] = referenceFrame[
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
        ] = referenceFrame[
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
        ] = referenceFrame[
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
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    # if no out of bounds
    targetFrame[
    startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
    startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
    ] = referenceFrame[
        startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
        startingRefPixel[1]:startingRefPixel[1] + macroblockSize
        ]
    return targetFrame
