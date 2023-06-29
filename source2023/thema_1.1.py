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

    seqErrorImages = []

    for P in range(1, len(frames)):
        # Calculate the error image of the current frame
        errorImage = calculateErrorImage(frames[P], frames[P - 1])

        # Add the error image to the error frames list
        seqErrorImages.append(errorImage)

    print(f'error frames: {seqErrorImages}')
    # Calculate the entropy of the error frames sequence
    H = entropy_score(seqErrorImages)

    print("Entropy score:", H)






