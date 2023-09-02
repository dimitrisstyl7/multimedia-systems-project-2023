from collections import Counter
from heapq import *

import numpy as np


def createHuffmanTreeVector(motionVectors):
    """
        Create the Huffman tree for the motion vectors
    """
    # Flatten the motion vectors
    motionVectorsFlat = [value for mv in motionVectors for value in mv]

    # Convert the flattened motion vectors to tuples
    motionVectorsTuples = [tuple(mv) for mv in motionVectorsFlat]

    # Create a leaf node for each unique character and build a min heap of all leaf nodes
    heap = [[wt, [sym, ""]] for sym, wt in Counter(motionVectorsTuples).items()]
    heapify(heap)

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

    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


def createHuffmanTableVector(huffmanTree):
    """
        Create the Huffman table
    """
    huffmanTable = {}
    for p in huffmanTree:
        huffmanTable[p[0]] = p[1]

    return huffmanTable


def encodeHuffmanVector(motionVectors, huffmanTable):
    """
        Encode the motion vectors
    """
    encodedMotionVectors = []
    for motionVector in motionVectors:
        encodedMotionVector = ''
        for value in motionVector:
            encodedMotionVector += huffmanTable[tuple(value)]
        encodedMotionVectors.append(encodedMotionVector)
    return encodedMotionVectors


def decodeHuffmanVector(encodedMotionVectors, huffmanTable, width, height):
    """
        Decode the motion vectors with the Huffman table
    """
    reverseTable = {code: symbol for symbol, code in huffmanTable.items()}
    decodedMotionVectors = []
    for encodedMotionVector in encodedMotionVectors:
        decodedMotionVector = []
        currentCode = ""
        for bit in encodedMotionVector:
            currentCode += bit
            if currentCode in reverseTable:
                decodedMotionVector.append(reverseTable[currentCode])
                currentCode = ""
        decodedMotionVector = np.array(decodedMotionVector, dtype='int').reshape((height, width))
        decodedMotionVectors.append(decodedMotionVector)
    return decodedMotionVectors
