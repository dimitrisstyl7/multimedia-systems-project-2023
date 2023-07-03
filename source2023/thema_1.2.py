from source2023.hierarchicalSearch import hierarchicalSearch
from source2023.videoFunction import *
from source2023.imageFunction import *
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

    # Create the Encoding Differential Pulse Code Modulation - DPCM
    for P in range(1, len(frames)):
        # Calculate the error image of the current frame
        errorImage = calculateErrorImage(frames[P], frames[P - 1])

        # Add the error image to the error frames list
        seqErrorImages.append(errorImage)

    seqErrorImages = np.array(seqErrorImages, dtype='uint8')

    print(seqErrorImages)

    videoSpecs = np.array([len(frames), width, height, fps], dtype='float64')

    print("Entropy of the original grayscale video is: ", entropy_score(originalFrames))

    # Calculate the entropy of the error frames sequence
    H = entropy_score(seqErrorImages)
    print("Entropy of the seqErrorFrames (grayscale) video is: ", H)

    # Create the video of the error frames sequence
    createVideoOutput(seqErrorImages, width, height, fps, 'thema_1_2_seqErrorFrames.avi')

    motion_vectors, motion_compensated_frames = hierarchicalSearch(frames, width, height)

    saveEncodedVideo(motion_vectors, 'thema_1_2_mV.pkl', motion_compensated_frames, 'thema_1_2_mCF.pkl', videoSpecs, 'thema_1_2_vS.pkl')

def videoDecoder():
    """
    Decode the video
    """
    pass

    motion_vectors, motion_compensated_frames, videoSpecs = readVideoInfo('thema_1_2_mV.pkl', 'thema_1_2_mCF.pkl', 'thema_1_2_vS.pkl')




if __name__ == '__main__':
    videoEncoder()
    # videoDecoder()