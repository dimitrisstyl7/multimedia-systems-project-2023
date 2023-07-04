import numpy as np


def motionCompensation(frames, motion_vectors):
    """
        Perform motion compensation on the frames.
    """
    for i in range(len(frames), 0, -1):
        # Get the previous and current frame
        curr_frame = frames[i]
        prev_frame = frames[i - 1]
        macroblock_size = 64
        starting_pixel = (i // macroblock_size, i % macroblock_size)  # (x, y)

        # Get the motion vector of the current macroblock, (x, y)
        motion_vector = motion_vectors[starting_pixel[0]][starting_pixel[1]]

        for row in range(starting_pixel[0], starting_pixel[0] + macroblock_size):
            for col in range(starting_pixel[1], starting_pixel[1] + macroblock_size):
                if motion_vector[0] < 0:  # x < 0
                    if motion_vector[1] < 0:  # y < 0
                        # Fill the previous frame with the current frame
                        prev_frame[row - macroblock_size:row, col - macroblock_size:col] = curr_frame[
                                                                                           row - macroblock_size:row,
                                                                                           col - macroblock_size:col]
                        pass

        # Perform motion compensation on the current frame
        motion_compensated_frame = motionCompensationOnFrame(i, motion_vectors, macroblock_size, width, height)

    return motion_compensated_frame


def motionCompensationOnFrame(curr_frame_macroblocks, motion_vectors, macroblock_size, width, height):
    """
        Perform motion compensation on the current frame.
    """
    # Create the motion compensated frame
    motion_compensated_frame = np.zeros((height, width), dtype='uint8')

    # For each macroblock in the current frame
    for i in range(0, len(curr_frame_macroblocks)):
        for j in range(0, len(curr_frame_macroblocks[i])):
            # Get the current macroblock
            curr_macroblock = curr_frame_macroblocks[i][j]

            # Get the motion vector of the current macroblock
            motion_vector = motion_vectors[i][j]

            # Get the x and y coordinates of the motion vector
            x = motion_vector[0]
            y = motion_vector[1]

            # Get the x and y coordinates of the current macroblock
            x1 = curr_macroblock[0]
            y1 = curr_macroblock[1]

            # Get the x and y coordinates of the current macroblock
            x2 = curr_macroblock[2]
            y2 = curr_macroblock[3]

            # Get the macroblock from the previous frame
            prev_macroblock = prev_frame[y1:y2, x1:x2]

            # Get the macroblock from the current frame
            curr_macroblock = curr_frame[y1:y2, x1:x2]

            # Get the motion compensated macroblock
            motion_compensated_macroblock = motionCompensationOnMacroblock(curr_macroblock, prev_macroblock, x, y,
                                                                           macroblock_size)

            # Add the motion compensated macroblock to the motion compensated frame
            motion_compensated_frame[y1:y2, x1:x2] = motion_compensated_macroblock

    return motion_compensated_frame
