import numpy as np

from progressBar import *

macroblockSize = 64


def motionCompensationForEncoding(frames, motionVectors, width, height):
    """
        Motion compensation for encoding.
    """
    noOfCols = width // macroblockSize
    motionCompensatedFrames = [frames[0]]  # I frame
    noOfMacroblocks = len(motionVectors[0])

    progressBar(0, len(frames), 'Creating Motion Compensation Frames: ', 'Motion Compensation Frames Created!')
    for i in range(1, len(frames)):
        # Get the reference (previous) and target (current) frame
        referenceFrame = frames[i - 1]
        targetFrame = np.zeros_like(referenceFrame)
        idxOfVectorsForCurrFrame = i - 1

        for j in range(noOfMacroblocks):
            # Get the motion vector of the current macroblock
            motionVector = motionVectors[idxOfVectorsForCurrFrame][j]

            # Get the starting pixel of the current macroblock
            macroblockIdx = (j // noOfCols, j % noOfCols)  # (y, x)
            startingRefPixel = (macroblockIdx[0] * macroblockSize, macroblockIdx[1] * macroblockSize)  # (y, x)
            targetFrame = motionCompensationForEncodingOnSpecificFrame(motionVector, targetFrame, referenceFrame,
                                                                       startingRefPixel, width, height)
        motionCompensatedFrames.append(targetFrame)
        progressBar(i + 1, len(frames), 'Creating Motion Compensation Frames: ', 'Motion Compensation Frames Created!')
    return motionCompensatedFrames


def motionCompensationForEncodingOnSpecificFrame(motionVector, targetFrame, referenceFrame, startingRefPixel, width,
                                                 height):
    """
        Motion compensation on a specific frame for encoding.
    """
    if motionVector[0] == 0 and motionVector[1] == 0:
        # No movement
        return targetFrame
    return performMotionCompensationForEncoding(startingRefPixel, motionVector, referenceFrame, targetFrame, width,
                                                height)


def performMotionCompensationForEncoding(startingRefPixel, motionVector, referenceFrame, targetFrame, width, height):
    """
        Perform motion compensation for encoding.
    """
    startingTargetPixel = (startingRefPixel[0] + motionVector[0], startingRefPixel[1] + motionVector[1])

    # out of bounds (top left)
    if startingTargetPixel[0] < 0 and startingTargetPixel[1] < 0:
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

    # out of bounds (bottom right)
    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and \
            startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        dy, dx = height - startingTargetPixel[0], width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]
        return targetFrame

    # out of bounds (top right)
    if startingTargetPixel[0] < 0 and startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
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

    # out of bounds (bottom left)
    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and startingTargetPixel[1] < 0:
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

    # out of bounds (left)
    if startingTargetPixel[1] < 0:
        dx = startingTargetPixel[1] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    # out of bounds (top)
    if startingTargetPixel[0] < 0:
        dy = startingTargetPixel[0] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize
            ]
        return targetFrame

    # out of bounds (right)
    if startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        dx = width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]
        return targetFrame

    # out of bounds (bottom)
    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0]:
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


def motionCompensationForDecoding(motionVectors, width, height, decodedSeqErrorImages):
    """
        Motion compensation for decoding.
    """
    noOfCols = width // macroblockSize
    motionCompensatedFrames = []
    noOfMacroblocks = len(motionVectors[0])
    iFrame = decodedSeqErrorImages[0]

    for i in range(1, len(motionVectors) + 1):  # len(motionVectors) equals to the number of frames
        if i == 1:
            referenceFrame = iFrame
        else:
            referenceFrame = motionCompensatedFrames[-1]  # Reference frame is the last frame in the list
        targetFrame = np.zeros_like(referenceFrame)
        seqErrorImage = decodedSeqErrorImages[i - 1]
        for j in range(noOfMacroblocks):
            # Get the motion vector of the current macroblock
            if i == 1:  # I frame
                motionVector = motionVectors[0][j]
            else:
                motionVector = motionVectors[i - 1][j]

            # Get the starting pixel of the current macroblock
            macroblockIdx = (j // noOfCols, j % noOfCols)  # (y, x)
            startingRefPixel = (macroblockIdx[0] * macroblockSize, macroblockIdx[1] * macroblockSize)  # (y, x)
            targetFrame = motionCompensationForDecodingOnSpecificFrame(motionVector, targetFrame, referenceFrame,
                                                                       startingRefPixel, width, height, seqErrorImage)
        motionCompensatedFrames.append(targetFrame)
    return motionCompensatedFrames


def motionCompensationForDecodingOnSpecificFrame(motionVector, targetFrame, referenceFrame, startingRefPixel, width,
                                                 height, seqErrorImage):
    """
        Motion compensation on a specific frame for decoding.
    """
    if motionVector[0] == 0 and motionVector[1] == 0:
        # No movement
        return targetFrame
    return performMotionCompensationForDecoding(startingRefPixel, motionVector, referenceFrame, targetFrame, width,
                                                height, seqErrorImage)


def performMotionCompensationForDecoding(startingRefPixel, motionVector, referenceFrame, targetFrame, width, height,
                                         seqErrorImage):
    """
        Perform motion compensation for decoding.
    """
    startingTargetPixel = (startingRefPixel[0] + motionVector[0], startingRefPixel[1] + motionVector[1])

    # out of bounds (top left)
    if startingTargetPixel[0] < 0 and startingTargetPixel[1] < 0:
        dy = startingTargetPixel[0] + macroblockSize
        dx = startingTargetPixel[1] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
             startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
             startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
             ]
        return targetFrame

    # out of bounds (bottom right)
    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and \
            startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        dy, dx = height - startingTargetPixel[0], width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
             startingRefPixel[0]:startingRefPixel[0] + dy,
             startingRefPixel[1]:startingRefPixel[1] + dx
             ]
        return targetFrame

    # out of bounds (top right)
    if startingTargetPixel[0] < 0 and startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        dy = startingTargetPixel[0] + macroblockSize
        dx = width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
             startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
             startingRefPixel[1]:startingRefPixel[1] + dx
             ]
        return targetFrame

    # out of bounds (bottom left)
    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0] and startingTargetPixel[1] < 0:
        dy = height - startingTargetPixel[0]
        dx = startingTargetPixel[1] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
             startingRefPixel[0]:startingRefPixel[0] + dy,
             startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
             ]
        return targetFrame

    # out of bounds (left)
    if startingTargetPixel[1] < 0:
        dx = startingTargetPixel[1] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
             startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
             startingRefPixel[1] + macroblockSize - dx:startingRefPixel[1] + macroblockSize
             ]
        return targetFrame

    # out of bounds (top)
    if startingTargetPixel[0] < 0:
        dy = startingTargetPixel[0] + macroblockSize
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
             startingRefPixel[0] + macroblockSize - dy:startingRefPixel[0] + macroblockSize,
             startingRefPixel[1]:startingRefPixel[1] + macroblockSize
             ]
        return targetFrame

    # out of bounds (right)
    if startingTargetPixel[1] + macroblockSize > targetFrame.shape[1]:
        dx = width - startingTargetPixel[1]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
            startingRefPixel[1]:startingRefPixel[1] + dx
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
             startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
             startingRefPixel[1]:startingRefPixel[1] + dx
             ]
        return targetFrame

    # out of bounds (bottom)
    if startingTargetPixel[0] + macroblockSize > targetFrame.shape[0]:
        dy = height - startingTargetPixel[0]
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] = referenceFrame[
            startingRefPixel[0]:startingRefPixel[0] + dy,
            startingRefPixel[1]:startingRefPixel[1] + macroblockSize
            ]

        # Add the error image to the target frame
        targetFrame[
        startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
        startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
        ] += seqErrorImage[
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

    # Add the error image to the target frame
    targetFrame[
    startingTargetPixel[0]:startingTargetPixel[0] + macroblockSize,
    startingTargetPixel[1]:startingTargetPixel[1] + macroblockSize
    ] += seqErrorImage[
         startingRefPixel[0]:startingRefPixel[0] + macroblockSize,
         startingRefPixel[1]:startingRefPixel[1] + macroblockSize
         ]
    return targetFrame
