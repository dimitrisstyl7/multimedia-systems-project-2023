import imageio
import pickle
import numpy as np
from scipy.stats import entropy

from source2023.imageFunction import convertToUint8


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
        writer.append_data(convertToUint8(frame))
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


# HuffmanTree
def huffmanTree(symbols):
    """
    Create the Huffman tree
    """
    tree = symbols.copy()
    while len(tree) > 1:
        # Sort the tree by the probabilities
        tree = tree[tree[:, 1].argsort()[::-1]]
        # Create the new node
        newNode = np.array([tree[-1, 0] + tree[-2, 0], tree[-1, 1] + tree[-2, 1]], dtype='object')
        # Add the new node to the tree
        tree = np.append(tree, [newNode], axis=0)
        # Remove the two last nodes
        tree = np.delete(tree, -3, axis=0)
        # Sort the tree by the probabilities
        tree = tree[tree[:, 1].argsort()[::-1]]
    return tree


# HuffmanCodebook
def huffmanCodebook(tree):
    """
    Create the Huffman codebook
    """
    codebook = {}
    for i in range(len(tree)):
        # Create the code
        code = ''
        node = tree[i]
        while node[2] != -1:
            # Add the bit to the code
            code = str(node[2]) + code
            # Go to the parent node
            node = tree[node[2]]
        # Add the code to the codebook
        codebook[tree[i][0]] = code
    return codebook

def huffmanEncode(image, codebook):
    """
    Encode the image using the Huffman codebook
    """
    encodedImage = ''
    for row in image:
        for pixel in row:
            encodedImage += codebook[pixel]
    return encodedImage
