from collections import Counter
from heapq import *

import numpy as np

from progressBar import progressBar


def createHuffmanTree(seqErrorImages):
    """
        Create the Huffman tree
    """
    # Create a leaf node for each unique character and build a min heap of all leaf nodes
    heap = [[wt, [sym, ""]] for sym, wt in Counter(seqErrorImages.flatten()).items()]
    heapify(heap)  # Transform list x into a heap, in-place, in linear time.

    while len(heap) > 1:
        # Extract two nodes with the minimum frequency from the min heap
        left = heappop(heap)
        right = heappop(heap)

        # For each extracted node, assign a bit 0 for left and 1 for the right
        for pair in left[1:]:
            pair[1] = '0' + pair[1]
        for pair in right[1:]:
            pair[1] = '1' + pair[1]

        # Create a new internal node with a frequency equal to the sum of the two nodes frequencies
        # Assign the left and right nodes as children of this new node
        heappush(heap, [left[0] + right[0]] + left[1:] + right[1:])

        # Return the root node
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))  # Return the root node


def createHuffmanTable(huffmanTree):
    """
        Create the Huffman table
    """
    huffmanTable = {}
    for p in huffmanTree:
        huffmanTable[p[0]] = p[1]

    return huffmanTable


def encodeHuffman(seqErrorImages, huffmanTable):
    """
        Encode the error frames sequence
    """
    encodedSeqErrorImages = []
    progressBar(0, len(seqErrorImages), 'Encoding the error frames sequence:', 'Encoded the error frames sequence!')
    for i, errorImage in enumerate(seqErrorImages):
        encodedSeqErrorImage = ''
        for pixel in errorImage.flatten():
            encodedSeqErrorImage += huffmanTable[pixel]
        encodedSeqErrorImages.append(encodedSeqErrorImage)
        progressBar(i + 1, len(seqErrorImages), 'Encoding the error frames sequence:',
                    'Encoded the error frames sequence!')
    return encodedSeqErrorImages


def decodeHuffman(encodedSeqErrorImages, huffmanTable, width, height):
    """
        Decode the error frames sequence with the Huffman table
    """
    reverseTable = {code: symbol for symbol, code in huffmanTable.items()}
    decodedSeqErrorImages = []
    total_frames = len(encodedSeqErrorImages)
    progressBar(0, total_frames, 'Decoding the error frames sequence:', 'Decoded the error frames sequence!')
    for i, encodedErrorImage in enumerate(encodedSeqErrorImages):
        decodedErrorImage = []
        currentCode = ""
        for bit in encodedErrorImage:
            currentCode += bit
            if currentCode in reverseTable:
                decodedErrorImage.append(reverseTable[currentCode])
                currentCode = ""
        decodedErrorImage = np.array(decodedErrorImage, dtype='uint8').reshape((height, width))
        decodedSeqErrorImages.append(np.array(decodedErrorImage))
        progressBar(i + 1, total_frames, 'Decoding the error frames sequence:', 'Decoded the error frames sequence!')
    return decodedSeqErrorImages
