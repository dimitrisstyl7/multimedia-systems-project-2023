macroblockSize = 64

def motionCompensationForEncoding(frames, motionVectors):
    """
        Perform motion compensation on the frames.
    """
    motionCompensatedFrames = [frames[0]]
    for i in range(1, len(frames)):
        # Get the previous and current frame
        currFrame = frames[i]
        prevFrame = frames[i - 1]
        startingPixel = (i // macroblockSize, i % macroblockSize)  # (x, y)

        # Get the motion vector of the current macroblock, (x, y)
        motionVector = motionVectors[startingPixel[0]][startingPixel[1]]

        if motionVector[0] < 0:  # x < 0
            if motionVector[1] < 0:  # y < 0
                # Move right and down
                prevFrame[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]:startingPixel[0] + macroblockSize
                ] = currFrame[
                    startingPixel[1]:startingPixel[1] + macroblockSize,
                    startingPixel[0]:startingPixel[0] + macroblockSize
                    ]
            if motionVector[1] > 0:  # y > 0
                # Move right and up
                prevFrame[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]:startingPixel[0] + macroblockSize
                ] = currFrame[
                    startingPixel[1]:startingPixel[1] - macroblockSize,
                    startingPixel[0]:startingPixel[0] + macroblockSize
                    ]
            else:  # y == 0
                # Move right
                prevFrame[
                startingPixel[1],
                startingPixel[0]:startingPixel[0] + macroblockSize
                ] = currFrame[
                    startingPixel[1],
                    startingPixel[0]:startingPixel[0] + macroblockSize
                    ]
        elif motionVector[0] > 0:  # x > 0
            if motionVector[1] < 0:
                # Move left and down
                prevFrame[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]:startingPixel[0] - macroblockSize
                ] = currFrame[
                    startingPixel[1]:startingPixel[1] + macroblockSize,
                    startingPixel[0]:startingPixel[0] - macroblockSize
                    ]
            if motionVector[1] > 0:
                # Move left and up
                prevFrame[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]:startingPixel[0] - macroblockSize
                ] = currFrame[
                    startingPixel[1]:startingPixel[1] - macroblockSize,
                    startingPixel[0]:startingPixel[0] - macroblockSize
                    ]
            else:  # y == 0
                # Move left
                prevFrame[
                startingPixel[1],
                startingPixel[0]:startingPixel[0] - macroblockSize
                ] = currFrame[
                    startingPixel[1],
                    startingPixel[0]:startingPixel[0] - macroblockSize
                    ]
        else:  # x == 0
            if motionVector[1] < 0:
                # Move down
                prevFrame[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]
                ] = currFrame[
                    startingPixel[1]:startingPixel[1] + macroblockSize,
                    startingPixel[0]
                    ]
            if motionVector[1] > 0:
                # Move up
                prevFrame[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]
                ] = currFrame[
                    startingPixel[1]:startingPixel[1] - macroblockSize,
                    startingPixel[0]
                    ]
            else:  # y == 0
                # No movement
                prevFrame = currFrame
        motionCompensatedFrames.append(prevFrame)
    return motionCompensatedFrames


def motionCompensationForDecoding(iFrame, motionVectors):
    """
        Perform motion compensation on the frames.
    """
    motionCompensatedFrames = [iFrame]
    for i in range(0, len(motionVectors)):
        # Get the previous and current frame
        prevFrame = motionCompensatedFrames[-1]
        startingPixel = (i // macroblockSize, i % macroblockSize)  # (x, y)

        # Get the motion vector of the current macroblock, (x, y)
        motionVector = motionVectors[startingPixel[0]][startingPixel[1]]

        if motionVector[0] < 0:  # x < 0
            if motionVector[1] < 0:  # y < 0
                # Move right and down
                prevFrame[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]:startingPixel[0] + macroblockSize
                ] = curr_frame[
                    startingPixel[1]:startingPixel[1] + macroblockSize,
                    startingPixel[0]:startingPixel[0] + macroblockSize
                    ]
            if motionVector[1] > 0:  # y > 0
                # Move right and up
                prevFrame[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]:startingPixel[0] + macroblockSize
                ] = curr_frame[
                    startingPixel[1]:startingPixel[1] - macroblockSize,
                    startingPixel[0]:startingPixel[0] + macroblockSize
                    ]
            else:  # y == 0
                # Move right
                prevFrame[
                startingPixel[1],
                startingPixel[0]:startingPixel[0] + macroblockSize
                ] = curr_frame[
                    startingPixel[1],
                    startingPixel[0]:startingPixel[0] + macroblockSize
                    ]
        elif motionVector[0] > 0:  # x > 0
            if motionVector[1] < 0:
                # Move left and down
                prevFrame[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]:startingPixel[0] - macroblockSize
                ] = curr_frame[
                    startingPixel[1]:startingPixel[1] + macroblockSize,
                    startingPixel[0]:startingPixel[0] - macroblockSize
                    ]
            if motionVector[1] > 0:
                # Move left and up
                prevFrame[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]:startingPixel[0] - macroblockSize
                ] = curr_frame[
                    startingPixel[1]:startingPixel[1] - macroblockSize,
                    startingPixel[0]:startingPixel[0] - macroblockSize
                    ]
            else:  # y == 0
                # Move left
                prevFrame[
                startingPixel[1],
                startingPixel[0]:startingPixel[0] - macroblockSize
                ] = curr_frame[
                    startingPixel[1],
                    startingPixel[0]:startingPixel[0] - macroblockSize
                    ]
        else:  # x == 0
            if motionVector[1] < 0:
                # Move down
                prevFrame[
                startingPixel[1]:startingPixel[1] + macroblockSize,
                startingPixel[0]
                ] = curr_frame[
                    startingPixel[1]:startingPixel[1] + macroblockSize,
                    startingPixel[0]
                    ]
            if motionVector[1] > 0:
                # Move up
                prevFrame[
                startingPixel[1]:startingPixel[1] - macroblockSize,
                startingPixel[0]
                ] = curr_frame[
                    startingPixel[1]:startingPixel[1] - macroblockSize,
                    startingPixel[0]
                    ]
            else:  # y == 0
                # No movement
                prevFrame = curr_frame
        motionCompensatedFrames.append(prevFrame)
    return motionCompensatedFrames
