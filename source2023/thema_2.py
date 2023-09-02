from videoFunction import *

videoPath = '../auxiliary2023/OriginalVideos/thema_2.avi'

def videoEncoder():
    """
            Encode the video
        """
    # ------------------------------- Load Video Properties -------------------------------- #
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
    createVideoOutput(frames, width, height, fps, 'thema_1_2_originalGrayScaleVideo.avi')
    print('Original grayscale video exported successfully!')


if __name__ == '__main__':
    entropy1 = videoEncoder()
    #entropy2 = videoDecoder()