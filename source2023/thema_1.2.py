from hierarchicalSearch import hierarchicalSearch
from motionCompensation import motionCompensation
from videoFunction import *

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"


def videoEncoder():
    """
        Encode the video
    """
    # Read the video
    frames, video_properties = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {video_properties[2]} pixels, a width of'
        f' {video_properties[1]} pixels and a framerate of {video_properties[3]} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    width = video_properties[1]
    height = video_properties[2]
    fps = video_properties[3]

    # The grayscale original video
    createVideoOutput(frames, width, height, fps, 'thema_1_2_originalGrayScaleVideo.avi')
    print("Original grayscale video exported successfully!")

    # Calculate the motion vectors using the hierarchical search algorithm
    motion_vectors = hierarchicalSearch(frames, width, height)

    # Calculate the motion compensated frames
    motion_compensated_frames = motionCompensation(frames, motion_vectors)

    seqErrorImages = calculateSeqErrorImages(frames)

    videoSpecs = np.array([len(frames), width, height, fps], dtype='float64')

    # Add all the frames to the original frames list
    originalFrames = [frames]
    H = entropy_score(originalFrames)
    print("Entropy of the original grayscale video is: ", H)

    # Create the video of the error frames sequence
    createVideoOutput(seqErrorImages, width, height, fps, 'thema_1_2_seqErrorFrames.avi')

    # motion_vectors, motion_compensated_frames = hierarchicalSearch(frames, width, height)
    #
    # saveEncodedVideo(motion_vectors, 'thema_1_2_mV.pkl', motion_compensated_frames, 'thema_1_2_mCF.pkl', videoSpecs,
    #                  'thema_1_2_vS.pkl')


def videoDecoder():
    """
        Decode the video
    """
    pass

    motion_vectors, motion_compensated_frames, videoSpecs = readVideoInfo('thema_1_2_mV.pkl', 'thema_1_2_mCF.pkl',
                                                                          'thema_1_2_vS.pkl')


if __name__ == '__main__':
    videoEncoder()
    # videoDecoder()
