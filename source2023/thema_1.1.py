import imageio
from source2023.videoFunctions import *
import numpy as np

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"

if __name__ == '__main__':
    # Read the video
    frames, fps = openVideo(videoPath)
    print(f'The video has {len(frames)} frames, a height of {frames[0].shape[0]} pixels, a width of {frames[0].shape[1]} pixels and a framerate of {fps} frames per second.')

    print(f'Frame before grayscale {frames[0]}')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    print(f'Frame after grayscale {frames[0]}')

    # print the first frame of the video as a jpg image in the source2023 folder
    image = Image.fromarray(frames[0].astype(np.uint8))
    image.save("firstFrame.jpg")