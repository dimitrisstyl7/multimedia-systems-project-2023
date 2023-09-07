from hierarchicalSearch import hierarchicalSearch
from motionCompensation_thema_2 import motionCompensation
from videoFunction import *

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
    # motionVectors = hierarchicalSearch(frames)
    # saveEncodedData(motionVectors, 'mv.pkl')
    motionVectors = readEncodedData('mv.pkl')

    # Calculate the motion compensated frames
    motionCompensatedFrames = motionCompensation(frames, motionVectors, width, height)
    createVideoOutput(motionCompensatedFrames, width, height, fps, 'thema_2_final.avi')
    print('Item disappeared successfully!')


if __name__ == '__main__':
    algorithm()
