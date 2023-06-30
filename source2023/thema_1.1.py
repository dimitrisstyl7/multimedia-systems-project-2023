from source2023.videoFunction import *
from source2023.imageFunction import *
import numpy as np

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"


def videoEncoder():
    # Read the video
    frames, fps = openVideo(videoPath)
    print(f'The video has {len(frames)} frames, a height of {frames[0].shape[0]} pixels, a width of {frames[0].shape[1]} pixels and a framerate of {fps} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    # The grayscale original video
    createVideoOutput(frames, fps, 'thema_1_1_originalGrayScaleVideo.avi')

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

    seqErrorImages = np.array(seqErrorImages, dtype='uint8')
    videoSpecs = np.array([len(frames), frames[0].shape[0], frames[0].shape[1], fps], dtype='int64')

    print("Entropy of the original video is: ", entropy_score(originalFrames))

    # Calculate the entropy of the error frames sequence
    H = entropy_score(seqErrorImages)
    print("Entropy of the seqErrorFrames (grayscale) video is: ", H)

    # Create the video of the error frames sequence
    createVideoOutput(seqErrorImages, fps, 'thema_1_1_seqErrorFrames.avi')

    # To help the decoder we will save the video properties
    saveVideoInfo(seqErrorImages, videoSpecs)


def videoDecoder():
    framesNumber, videoSpecs = readVideoInfo()

    print(f'The video has {videoSpecs[0]} frames, a height of {videoSpecs[1]} pixels, a width of {videoSpecs[2]} pixels and a framerate of {videoSpecs[3]} frames per second.')

    frames = np.reshape(framesNumber, (videoSpecs[0], videoSpecs[1], videoSpecs[2]))

    # Create the output video
    createVideoOutput(frames, videoSpecs[3], 'thema_1_1_decodedVideo.avi')

    first_frame_flag = True

    decodedFrames = []

    prev_frame = None

    for frame in frames:

        # If it's the first frame (which is the original - no error) then write it to file, make it the previous
        # frame and set the flag to false because we found the first frame
        #
        # Else add the previous frame to the current on, write it in the file
        # and set it as the previous frame because we used it
        if first_frame_flag:
            decodedFrames.append(frame)
            prev_frame = frame
            first_frame_flag = False
        else:
            decodedFrame = np.add(prev_frame, frame)
            decodedFrames.append(frame)
            prev_frame = decodedFrame

    decodedFrames = np.array(decodedFrames, dtype='uint8')

    createVideoOutput(decodedFrames, videoSpecs[3], 'thema_1_1_decodedVideo.avi')


if __name__ == '__main__':
    videoEncoder()
    videoDecoder()
