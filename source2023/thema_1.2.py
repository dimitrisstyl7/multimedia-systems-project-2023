from source2023.videoFunction import *
import numpy as np

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"

def videoEncoder():
    """
    Encode the video
    """
    # Read the video
    frames, video_properties = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {video_properties[2]} pixels, a width of {video_properties[1]} pixels and a framerate of {video_properties[3]} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    width = video_properties[1]
    height = video_properties[2]
    fps = video_properties[3]

    # The grayscale original video
    createVideoOutput(frames, width, height, fps, 'thema_1_2_originalGrayScaleVideo.avi')

    originalFrames = []
    seqErrorImages = []

    # Add all the frames to the original frames list
    originalFrames.append(frames)

    # Add the first frame to the error frames list (I frame)
    seqErrorImages.append(frames[0])



if __name__ == '__main__':
    videoEncoder()