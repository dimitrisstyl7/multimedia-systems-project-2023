import imageio
import numpy as np
from scipy.stats import entropy



def openVideo(file):
    video = imageio.get_reader(file, 'ffmpeg')
    fps = video.get_meta_data()['fps']
    i = 0  # Frame counter
    frames = []  # List of frames
    while True:
        try:
            frames.append(video.get_data(i))
            i += 1
        except IndexError:
            break

    return np.array(frames), fps


def createVideoOutput(frames, fps, name):
    writer = imageio.get_writer("../auxiliary2023/OutputVideos/" + name, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

# Convert RGB image to grayscale
def rgb2gray(rgb):
    # The ITU-R BT.709 (HDTV) standard for converting RGB to grayscale
    return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])


# Create video to grayscale
def createGrayscaleVideo(frames):
    grayscaleFrames = []
    for frame in frames:
        grayscaleFrames.append(rgb2gray(frame))
    return np.array(grayscaleFrames)


def entropy_score(error_frames):
    values, counts = np.unique(error_frames, return_counts=True) # values: unique values of error_frames, counts: how many times each value appears
    return entropy(counts)
