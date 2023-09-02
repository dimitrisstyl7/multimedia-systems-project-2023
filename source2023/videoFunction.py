import pickle

import cv2
import numpy as np
from scipy.stats import entropy

from imageFunction import calculateErrorImage
from progressBar import progressBar


def openVideo(file):
    """
        Open the video and return the frames and the video properties
    """
    video = cv2.VideoCapture(file)
    framesNum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    videoWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_properties = [framesNum, videoWidth, videoHeight, fps]
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return np.array(frames), video_properties


def createVideoOutput(frames, width, height, fps, name):
    """
        Create the video output
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter("../auxiliary2023/OutputVideos/" + name, fourcc, fps, (width, height))
    for frame in frames:
        gray_color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        videoWriter.write(gray_color_frame)
    videoWriter.release()


# Create video to grayscale
def createGrayscaleVideo(frames):
    """
        Convert the video to grayscale
    """
    grayscaleFrames = []
    for frame in frames:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscaleFrames.append(grayFrame)
    return np.array(grayscaleFrames)


def calSeqErrorImages(frames):
    """
        Calculate the error frames sequence
    """
    # Add the first frame to the error frames list (I frame)
    seqErrorImages = [frames[0]]

    # Create the Encoding Differential Pulse Code Modulation - DPCM
    for P in range(1, len(frames)):
        # Calculate the error image of the current frame
        errorImage = calculateErrorImage(frames[P], frames[P - 1])

        # Add the error image to the error frames list
        seqErrorImages.append(errorImage)

    return np.array(seqErrorImages, dtype='uint8')


def calculateSeqErrorImages(originalFrames, motionCompensatedFrames):
    """
        Calculate the error frames sequence
    """
    # Add the first frame to the error frames list (I frame)
    seqErrorImages = [originalFrames[0]]

    # Create the Encoding Differential Pulse Code Modulation - DPCM
    progressBar(0, len(originalFrames), 'Creating Sequence Error Images: ', 'Sequence Error Images Created!')

    for P in range(1, len(originalFrames)):
        # Calculate the error image of the current frame
        errorImage = calculateErrorImage(originalFrames[P], motionCompensatedFrames[P])

        # Add the error image to the error frames list
        seqErrorImages.append(errorImage)
        progressBar(P + 1, len(originalFrames), 'Creating Sequence Error Images: ', 'Sequence Error Images Created!')

    return np.array(seqErrorImages, dtype='uint8')


def addSeqErrorImagesToCompensatedFrames(motionCompensatedFrames, seqErrorImages):
    """
        Add the error frames sequence to the motion compensated frames.
    """
    outputFrames = [seqErrorImages[0]]
    for i in range(0, len(motionCompensatedFrames)):
        seqErrorFrame = seqErrorImages[i + 1]
        motionCompensatedFrame = motionCompensatedFrames[i]
        outputFrame = motionCompensatedFrame + seqErrorFrame
        outputFrames.append(outputFrame)
    return outputFrames


def entropyScore(errorFrames):
    """
        Calculate the entropy of the error frames sequence
    """
    # values: unique values of error_frames, counts: how many times each value appears
    values, counts = np.unique(errorFrames, return_counts=True)
    return entropy(counts)


def saveEncodedData(data, fileName):
    """
        Save the video properties to a binary file
    """
    with open('../auxiliary2023/VideoProperties/' + fileName, 'wb') as file:
        pickle.dump(data, file)


def readEncodedData(fileName):
    """
        Read the video properties from a binary file
    """
    with open('../auxiliary2023/VideoProperties/' + fileName, 'rb') as file:
        data = pickle.load(file)
    return data
