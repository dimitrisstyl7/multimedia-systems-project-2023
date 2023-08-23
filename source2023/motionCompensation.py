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
        refFrame = frames[i - 1]
        targetFrame = np.zeros_like(refFrame)
        idxOfVectorsForCurrFrame = i - 1
        noOfMacroblocks = len(motionVectors[idxOfVectorsForCurrFrame])

        for j in range(noOfMacroblocks):
            # Get the motion vector of the current macroblock
            motionVector = motionVectors[idxOfVectorsForCurrFrame][j][0]

            # Get the starting pixel of the current macroblock
            macroblockIdx = (j % noOfCols, j // noOfCols)  # (x, y)
            startingPixel = (macroblockIdx[0] * macroblockSize, macroblockIdx[1] * macroblockSize)  # (x, y)
            targetFrame = motionCompensationForSpecificFrame(motionVector, targetFrame, refFrame, startingPixel)

        motionCompensatedFrames.append(targetFrame)
    return motionCompensatedFrames


def motionCompensationForDecoding(iFrame, motionVectors):
    """
        Motion compensation for decoding.
    """
    motionCompensatedFrames = [iFrame]
    for i in range(len(motionVectors)):  # len(motionVectors) means the number of frames
        prevFrame = motionCompensatedFrames[-1]  # Previous frame is the last frame in the list
        currFrame = prevFrame.copy()  # Current frame is the prevFrame and we will apply the motion compensation on it
        for j in range(len(motionVectors[i])):  # len(motionVectors[i]) means the number of macroblocks in the frame
            startingPixel = (j // macroblockSize, j % macroblockSize)  # (x, y)
            motionVector = motionVectors[i][j]  # Get the motion vector of the current macroblock, (x, y)
        motionCompensatedFrames.append(
            motionCompensationForSpecificFrame(motionVector, currFrame, prevFrame, startingPixel))
    return motionCompensatedFrames


def motionCompensationForSpecificFrame(motionVector, frame1, frame2, startingPixel):
    """
        Perform motion compensation on a specific frame.
    """
    if motionVector[0] < 0:  # x < 0
        if motionVector[1] < 0:  # y < 0
            # Move right and down
            frame1[
            startingPixel[1]:startingPixel[1] + macroblockSize,
            startingPixel[0]:startingPixel[0] + macroblockSize
            ] = frame2[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]:startingPixel[0] + macroblockSize
                ]
        if motionVector[1] > 0:  # y > 0
            # Move right and up
            frame1[
            startingPixel[1]:startingPixel[1] - macroblockSize,
            startingPixel[0]:startingPixel[0] + macroblockSize
            ] = frame2[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]:startingPixel[0] + macroblockSize
                ]
        else:  # y == 0
            # Move right
            frame1[
            startingPixel[1],
            startingPixel[0]:startingPixel[0] + macroblockSize
            ] = frame2[
                startingPixel[1],
                startingPixel[0]:startingPixel[0] + macroblockSize
                ]
    elif motionVector[0] > 0:  # x > 0
        if motionVector[1] < 0:  # y < 0
            # Move left and down
            frame1[
            startingPixel[1]:startingPixel[1] + macroblockSize,
            startingPixel[0]:startingPixel[0] - macroblockSize
            ] = frame2[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]:startingPixel[0] - macroblockSize
                ]
        if motionVector[1] > 0:  # y > 0
            # Move left and up
            frame1[
            startingPixel[1]:startingPixel[1] - macroblockSize,
            startingPixel[0]:startingPixel[0] - macroblockSize
            ] = frame2[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]:startingPixel[0] - macroblockSize
                ]
        else:  # y == 0
            # Move left
            frame1[
            startingPixel[1],
            startingPixel[0]:startingPixel[0] - macroblockSize
            ] = frame2[
                startingPixel[1],
                startingPixel[0]:startingPixel[0] - macroblockSize
                ]
    else:  # x == 0
        if motionVector[1] < 0:  # y < 0
            # Move down
            frame1[
            startingPixel[1]:startingPixel[1] + macroblockSize,
            startingPixel[0]
            ] = frame2[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]
                ]
        if motionVector[1] > 0:  # y > 0
            # Move up
            frame1[
            startingPixel[1]:startingPixel[1] - macroblockSize,
            startingPixel[0]
            ] = frame2[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]
                ]
        else:  # y == 0
            # No movement
            frame1 = frame2
    return frame1
