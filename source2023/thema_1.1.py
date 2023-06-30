from source2023.videoFunction import *
from source2023.imageFunction import *
import numpy as np

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"

if __name__ == '__main__':
    # Read the video
    frames, fps = openVideo(videoPath)
    print(f'The video has {len(frames)} frames, a height of {frames[0].shape[0]} pixels, a width of {frames[0].shape[1]} pixels and a framerate of {fps} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    originalFrames = []
    seqErrorImages = []

    # Add the first frame to the original frames list
    originalFrames.append(convertToUint8(frames[0]))

    # Add the first frame to the error frames list (I frame)
    seqErrorImages.append(convertToUint8(frames[0]))

    for P in range(1, len(frames)):
        # Calculate the error image of the current frame
        errorImage = calculateErrorImage(frames[P], frames[P - 1])

        # Add the error image to the error frames list
        seqErrorImages.append(convertToUint8(errorImage))

    print(len(seqErrorImages))

    # Calculate the entropy of the error frames sequence
    H = entropy_score(seqErrorImages)

    print("Entropy of the original (grayscale) video is: ", H)

    # Create the video of the error frames sequence
    createVideoOutput(seqErrorImages, fps, 'thema_1_1.avi')






