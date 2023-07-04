from collections import Counter
from heapq import *
import numpy as np


# Create the Huffman tree for the error frames sequence
def createHuffmanTreeVector(seqErrorImages):
    """
        Create the Huffman tree
    """
    # Flatten the error frames sequence
    seqErrorImagesFlat = [pixel for frame in seqErrorImages for pixel in frame]

    # Convert the flattened sequence to tuples
    seqErrorImagesTuples = [tuple(pixel) for pixel in seqErrorImagesFlat]

    # Create a leaf node for each unique character and build a min heap of all leaf nodes
    heap = [[wt, [sym, ""]] for sym, wt in Counter(seqErrorImagesTuples).items()]
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


# Create the Huffman table
def createHuffmanTableVector(huffmanTree):
    """
        Create the Huffman table
    """
    huffmanTable = {}
    for p in huffmanTree:
        huffmanTable[p[0]] = p[1]

    return huffmanTable


# Encode the error frames sequence
def encodeHuffmanVector(seqErrorImages, huffmanTable):
    """
        Encode the error frames sequence
    """
    encodedSeqErrorImages = []
    for errorImage in seqErrorImages:
        encodedSeqErrorImage = ''
        for pixel in errorImage:
            encodedSeqErrorImage += huffmanTable[tuple(pixel)]
        encodedSeqErrorImages.append(encodedSeqErrorImage)
    return encodedSeqErrorImages


# Decode the error frames sequence with the Huffman table
def decodeHuffmanVector(encodedSeqErrorImages, huffmanTable, width, height):
    """
        Decode the error frames sequence with the Huffman table
    """
    reverseTable = {code: symbol for symbol, code in huffmanTable.items()}
    decodedSeqErrorImages = []
    for encodedErrorImage in encodedSeqErrorImages:
        decodedErrorImage = []
        currentCode = ""
        for bit in encodedErrorImage:
            currentCode += bit
            if currentCode in reverseTable:
                decodedErrorImage.append(reverseTable[currentCode])
                currentCode = ""
        decodedErrorImage = np.array(decodedErrorImage, dtype='int').reshape((height, width))
        decodedSeqErrorImages.append(decodedErrorImage)
    return decodedSeqErrorImages

#
# # Test the methods with the provided input
# seqErrorImages = [[(1, -3), (2, 1), (0, 0)], [(1, -3), (2, 1), (2, 0)]]
#
# # Convert the tuples to NumPy array
# seqErrorImages_np = np.array(seqErrorImages)
#
# # Create Huffman tree
# huffmanTree = createHuffmanTree(seqErrorImages_np)
#
# # Create Huffman table
# huffmanTable = createHuffmanTable(huffmanTree)
#
# # Encode the error frames sequence
# encodedSeqErrorImages = encodeHuffman(seqErrorImages_np, huffmanTable)
# print("Encoded sequence:", encodedSeqErrorImages)
#
# # Decode the error frames sequence
# decodedSeqErrorImages = decodeHuffman(encodedSeqErrorImages, huffmanTable, width=3, height=2)
#
# # Convert the decoded sequence to the desired format
# decodedSeqErrorImages = [[tuple(pixel) for pixel in frame] for frame in decodedSeqErrorImages]
# print("Decoded sequence:")
# for errorImage in decodedSeqErrorImages:
#     print(errorImage)
