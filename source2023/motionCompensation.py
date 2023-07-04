def motionCompensation(frames, motion_vectors):
    """
        Perform motion compensation on the frames.
    """
    motion_compensated_frames = []
    for i in range(1, len(frames)):
        # Get the previous and current frame
        curr_frame = frames[i]
        prev_frame = frames[i - 1]
        macroblock_size = 64
        starting_pixel = (i // macroblock_size, i % macroblock_size)  # (x, y)

        # Get the motion vector of the current macroblock, (x, y)
        motion_vector = motion_vectors[starting_pixel[0]][starting_pixel[1]]

        if motion_vector[0] < 0:  # x < 0
            if motion_vector[1] < 0:  # y < 0
                # Move right and down
                prev_frame[
                starting_pixel[1]:starting_pixel[1] + macroblock_size,
                starting_pixel[0]:starting_pixel[0] + macroblock_size
                ] = curr_frame[
                    starting_pixel[1]:starting_pixel[1] + macroblock_size,
                    starting_pixel[0]:starting_pixel[0] + macroblock_size
                    ]
            if motion_vector[1] > 0:  # y > 0
                # Move right and up
                prev_frame[
                starting_pixel[1]:starting_pixel[1] - macroblock_size,
                starting_pixel[0]:starting_pixel[0] + macroblock_size
                ] = curr_frame[
                    starting_pixel[1]:starting_pixel[1] - macroblock_size,
                    starting_pixel[0]:starting_pixel[0] + macroblock_size
                    ]
            else:  # y == 0
                # Move right
                prev_frame[
                starting_pixel[1],
                starting_pixel[0]:starting_pixel[0] + macroblock_size
                ] = curr_frame[
                    starting_pixel[1],
                    starting_pixel[0]:starting_pixel[0] + macroblock_size
                    ]
        elif motion_vector[0] > 0:  # x > 0
            if motion_vector[1] < 0:
                # Move left and down
                prev_frame[
                starting_pixel[1]:starting_pixel[1] + macroblock_size,
                starting_pixel[0]:starting_pixel[0] - macroblock_size
                ] = curr_frame[
                    starting_pixel[1]:starting_pixel[1] + macroblock_size,
                    starting_pixel[0]:starting_pixel[0] - macroblock_size
                    ]
            if motion_vector[1] > 0:
                # Move left and up
                prev_frame[
                starting_pixel[1]:starting_pixel[1] - macroblock_size,
                starting_pixel[0]:starting_pixel[0] - macroblock_size
                ] = curr_frame[
                    starting_pixel[1]:starting_pixel[1] - macroblock_size,
                    starting_pixel[0]:starting_pixel[0] - macroblock_size
                    ]
            else:  # y == 0
                # Move left
                prev_frame[
                starting_pixel[1],
                starting_pixel[0]:starting_pixel[0] - macroblock_size
                ] = curr_frame[
                    starting_pixel[1],
                    starting_pixel[0]:starting_pixel[0] - macroblock_size
                    ]
        else:  # x == 0
            if motion_vector[1] < 0:
                # Move down
                prev_frame[
                starting_pixel[1]:starting_pixel[1] + macroblock_size,
                starting_pixel[0]
                ] = curr_frame[
                    starting_pixel[1]:starting_pixel[1] + macroblock_size,
                    starting_pixel[0]
                    ]
            if motion_vector[1] > 0:
                # Move up
                prev_frame[
                starting_pixel[1]:starting_pixel[1] - macroblock_size,
                starting_pixel[0]
                ] = curr_frame[
                    starting_pixel[1]:starting_pixel[1] - macroblock_size,
                    starting_pixel[0]
                    ]
            else:  # y == 0
                # No movement
                prev_frame = curr_frame
        motion_compensated_frames.append(prev_frame)
    return motion_compensated_frames
