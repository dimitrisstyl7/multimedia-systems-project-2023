import pickle

import cv2
import numpy as np
from scipy.stats import entropy

from imageFunction import calculateErrorImage


def openVideo(file):
    """
        Open the video and return the frames and the video properties
    """
    video = cv2.VideoCapture(file)
    frames_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_properties = [frames_num, video_width, video_height, fps]
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
    video_writer = cv2.VideoWriter("../auxiliary2023/OutputVideos/" + name, fourcc, fps, (width, height))
    for frame in frames:
        gray_color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(gray_color_frame)
    video_writer.release()


# Create video to grayscale
def createGrayscaleVideo(frames):
    """
        Convert the video to grayscale
    """
    grayscaleFrames = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscaleFrames.append(gray_frame)
    return np.array(grayscaleFrames)


def calculateSeqErrorImages(frames):
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


def entropy_score(error_frames):
    """
        Calculate the entropy of the error frames sequence
    """
    # values: unique values of error_frames, counts: how many times each value appears
    values, counts = np.unique(error_frames, return_counts=True)
    return entropy(counts)


def saveEncodedVideo(data1, fileName1, data2, fileName2, data3, fileName3):
    """
        Save the video properties to a binary file
    """
    with open('../auxiliary2023/VideoProperties/' + fileName1, 'wb') as file:
        pickle.dump(data1, file)
    with open('../auxiliary2023/VideoProperties/' + fileName2, 'wb') as file:
        pickle.dump(data2, file)
    with open('../auxiliary2023/VideoProperties/' + fileName3, 'wb') as file:
        pickle.dump(data3, file)


def readVideoInfo(fileName1, fileName2, fileName3):
    """
        Read the video properties from a binary file
    """
    with open('../auxiliary2023/VideoProperties/' + fileName1, 'rb') as file:
        data1 = pickle.load(file)
    with open('../auxiliary2023/VideoProperties/' + fileName2, 'rb') as file:
        data2 = pickle.load(file)
    with open('../auxiliary2023/VideoProperties/' + fileName3, 'rb') as file:
        data3 = pickle.load(file)
    return data1, data2, data3
