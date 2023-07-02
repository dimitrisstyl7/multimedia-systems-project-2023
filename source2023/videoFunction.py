#import imageio
import cv2
import pickle
import numpy as np
from scipy.stats import entropy

from source2023.imageFunction import convertToUint8


def openVideo(file):
    video = cv2.VideoCapture(file)
    frames_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_properties = []
    video_properties.append(frames_num)
    video_properties.append(video_width)
    video_properties.append(video_height)
    video_properties.append(fps)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return np.array(frames), video_properties


def createVideoOutput(frames, video_properties, name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("../auxiliary2023/OutputVideos/" + name, fourcc, video_properties[3], (video_properties[1], video_properties[2]))
    for frame in frames:
        gray_color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(gray_color_frame)
    video_writer.release()
"""
# Convert RGB image to grayscale
def rgb2gray(rgb):
    # The ITU-R BT.709 (HDTV) standard for converting RGB to grayscale
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])"""


# Create video to grayscale
def createGrayscaleVideo(frames):
    grayscaleFrames = []
    for frame in frames:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscaleFrames.append(gray_frame)
    return np.array(grayscaleFrames)


def entropy_score(error_frames):
    # values: unique values of error_frames, counts: how many times each value appears
    values, counts = np.unique(error_frames, return_counts=True)
    return entropy(counts)


def saveVideoInfo(seqErrorImages, nameSeq, videoSpecs, nameSpecs):
    """
    Save the video properties to a binary file
    """
    with open('../auxiliary2023/VideoProperties/' + nameSeq, 'wb') as file:
        pickle.dump(seqErrorImages, file)
    with open('../auxiliary2023/VideoProperties/' + nameSpecs, 'wb') as file:
        pickle.dump(videoSpecs, file)


def readVideoInfo(nameSeq, nameSpecs):
    """
    Read the video properties from a binary file
    """
    with open('../auxiliary2023/VideoProperties/' + nameSeq, 'rb') as file:
        seqErrorImages = pickle.load(file)
    with open('../auxiliary2023/VideoProperties/' + nameSpecs, 'rb') as file:
        videoSpecs = pickle.load(file)
    return seqErrorImages, videoSpecs