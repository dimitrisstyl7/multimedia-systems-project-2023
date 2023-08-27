import numpy as np

macroblockSize = 64


def motionCompensationForEncoding(frames, motionVectors, width):
    """
        Motion compensation for encoding.
    """
    noOfCols = width // macroblockSize
    motionCompensatedFrames = [frames[0]]  # I frame

    for i in range(1, len(frames)):
        print(f'\nFrame {i} of {len(frames) - 1}')
        # Get the reference (previous) and target (current) frame
        referenceFrame = frames[i - 1]
        targetFrame = np.zeros_like(referenceFrame)
        idxOfVectorsForCurrFrame = i - 1
        noOfMacroblocks = len(motionVectors[idxOfVectorsForCurrFrame])

        for j in range(noOfMacroblocks):
            print(f'\tMacroblock {j} of {noOfMacroblocks - 1}')
            # Get the motion vector of the current macroblock
            motionVector = motionVectors[idxOfVectorsForCurrFrame][j][0]

            # Get the starting pixel of the current macroblock
            macroblockIdx = (j // noOfCols, j % noOfCols)  # (y, x)
            startingRefPixel = (macroblockIdx[0] * macroblockSize, macroblockIdx[1] * macroblockSize)  # (y, x)
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
            startingPixel = (j % macroblockSize, j // macroblockSize)  # (y, x)
            motionVector = motionVectors[i][j]  # Get the motion vector of the current macroblock, (y, x)
        motionCompensatedFrames.append(
            motionCompensationForSpecificFrame(motionVector, targetFrame, referenceFrame, startingPixel))
    return motionCompensatedFrames


def motionCompensationForEncodingOnSpecificFrame(motionVector, targetFrame, referenceFrame, startingRefPixel):
    """
        Perform motion compensation on a specific frame.
    """
    if motionVector[1] < 0:  # x < 0
        if motionVector[0] < 0:  # y < 0
            # Move left and up
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

        if motionVector[0] > 0:  # y > 0
            # Move left and down
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

        else:  # y == 0
            # Move left
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

    elif motionVector[1] > 0:  # x > 0
        if motionVector[0] < 0:  # y < 0
            # Move right and up
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

        if motionVector[0] > 0:  # y > 0
            # Move right and down
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

        else:  # y == 0
            # Move right
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

    else:  # x == 0
        if motionVector[0] < 0:  # y < 0
            # Move up
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

        if motionVector[0] > 0:  # y > 0
            # Move down
            performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame)

        else:  # y == 0
            # No movement
            targetFrame = referenceFrame

    return targetFrame


def performMotionCompensation(startingRefPixel, motionVector, referenceFrame, targetFrame):
    """
        Perform motion compensation.
    """
    startingTargetPixel = (startingRefPixel[0] + motionVector[0], startingRefPixel[1] + motionVector[1])

    if startingTargetPixel[0] < 0 and startingTargetPixel[1] < 0:
        # out of bounds (top left)
        ''' CODE HERE '''
        return targetFrame

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and \
            startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (bottom right)
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize - motionVector[0],
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize - motionVector[1]
            ]
        return targetFrame

    if startingTargetPixel[0] < 0 and startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (top right)
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            :startingRefPixel[0] + macroblockSize + motionVector[0],
            ]
        return targetFrame

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and startingTargetPixel[1] < 0:
        # out of bounds (bottom left)
        ''' CODE HERE '''
        return targetFrame

    if startingTargetPixel[1] < 0:
        # out of bounds (left)
        ''' CODE HERE '''
        return targetFrame

    if startingTargetPixel[0] < 0:
        # out of bounds (top)
        ''' CODE HERE '''
        return targetFrame

    if startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        # out of bounds (right)
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + motionVector[1]
            ]
        return targetFrame

    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0]:
        # out of bounds (bottom)
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + motionVector[0],
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
