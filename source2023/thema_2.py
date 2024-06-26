import os

from hierarchical_search import hierarchicalSearch
from motion_compensation_thema_2 import motionCompensation
from video_function import *

videoPath = '../auxiliary2023/OriginalVideos/thema_2.avi'


def algorithm():
    # Read the video
    frames, videoProperties = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {videoProperties[2]} pixels, a width of'
        f' {videoProperties[1]} pixels and a framerate of {videoProperties[3]} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    width = videoProperties[1]
    height = videoProperties[2]
    fps = videoProperties[3]

    # The grayscale original video
    createVideoOutput(frames, width, height, fps, 'thema_2_originalGrayScaleVideo.avi')
    print('Original grayscale video exported successfully!')

    # Calculate the motion vectors using the hierarchical search algorithm
    motionVectors = hierarchicalSearch(frames)

    # Calculate the motion compensated frames
    motionCompensatedFrames = motionCompensation(frames, motionVectors, width, height)
    createVideoOutput(motionCompensatedFrames, width, height, fps, 'thema_2_final.avi')
    print('Item disappeared successfully!')


def createFoldersIfNotExist():
    if not os.path.exists('../auxiliary2023/OriginalVideos'):
        os.makedirs('../auxiliary2023/OriginalVideos')
    if not os.path.exists('../auxiliary2023/OutputVideos'):
        os.makedirs('../auxiliary2023/OutputVideos')
    if not os.path.exists('../auxiliary2023/VideoProperties'):
        os.makedirs('../auxiliary2023/VideoProperties')


if __name__ == '__main__':
    createFoldersIfNotExist()
    algorithm()
