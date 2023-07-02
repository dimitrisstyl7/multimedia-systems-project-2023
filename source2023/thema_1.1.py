from source2023.videoFunction import *
from source2023.imageFunction import *
import numpy as np

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"


def videoEncoder():
    # Read the video
    frames, video_properties = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {video_properties[2]} pixels, a width of {video_properties[1]} pixels and a framerate of {video_properties[3]} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    # The grayscale original video
    createVideoOutput(frames, video_properties, 'thema_1_1_originalGrayScaleVideo.avi')

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
        seqErrorImages.append(convertToUint8(errorImage))

    seqErrorImages = np.array(seqErrorImages, dtype='uint8')
    #
    videoSpecs = np.array([len(frames), video_properties[2], video_properties[1], video_properties[3]], dtype='float64')

    print("Entropy of the original grayscale video is: ", entropy_score(originalFrames))

    # Calculate the entropy of the error frames sequence
    H = entropy_score(seqErrorImages)
    print("Entropy of the seqErrorFrames (grayscale) video is: ", H)

    # Create the video of the error frames sequence
    createVideoOutput(seqErrorImages, video_properties, 'thema_1_1_seqErrorFrames.avi')

    # To help the decoder we will save the video properties
    saveVideoInfo(seqErrorImages, 'thema_1_1_seqErrorFrames.pkl', videoSpecs, 'thema_1_1_videoSpecs.pkl')


def videoDecoder():
    framesNumber, videoSpecs = readVideoInfo('thema_1_1_seqErrorFrames.pkl', 'thema_1_1_videoSpecs.pkl')
    """
    num_frames = int(videoSpecs[0])
    height = int(videoSpecs[1])
    width = int(videoSpecs[2])
    loaded_fps = float(videoSpecs[3])"""

    video_properties = []
    video_properties.append(int(videoSpecs[0]))
    video_properties.append(int(videoSpecs[1]))
    video_properties.append(int(videoSpecs[2]))
    video_properties.append(float(videoSpecs[3]))

    print(f'The video has {video_properties[0]} frames, a height of {video_properties[1]} pixels, a width of {video_properties[2]} pixels and a framerate of {video_properties[3]} frames per second.')

    frames = np.reshape(framesNumber, (video_properties[0], video_properties[1], video_properties[2]))
    """
    # Create the output video
    createVideoOutput(frames, videoSpecs[3], 'thema_1_1_decodedVideo.avi')"""

    first_frame_flag = True

    decodedFrames = []

    prev_frame = None
    i = 0
    for frame in frames:
        if first_frame_flag:
            decodedFrames.append(frame)
            saveImage(frame, str(i) + '.jpg')
            prev_frame = frame
            first_frame_flag = False
        else:
            decodedFrame = prev_frame + frame  # Add frame to prev_frame
            saveImage(decodedFrame, str(i) + '.jpg')
            decodedFrames.append(decodedFrame)
            prev_frame = decodedFrame
        i += 1

    decodedFrames = np.array(decodedFrames, dtype='uint8')

    createVideoOutput(decodedFrames, video_properties, 'thema_1_1_decodedVideo.avi')


if __name__ == '__main__':
    videoEncoder()
    videoDecoder()
