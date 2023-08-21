from huffman import *
from huffmanVectors import *
from motionCompensation import *
from hierarchicalSearch import hierarchicalSearch
from videoFunction import *

videoPath = '../auxiliary2023/OriginalVideos/thema_1.avi'


def videoEncoder():
    """
        Encode the video
    """
    # ------------------------------- Load Video Properties -------------------------------- #
    # Read the video
    frames, videoProperties = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {videoProperties[2]} pixels, a width of'
        f' {videoProperties[1]} pixels and a framerate of {videoProperties[3]} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    width = videoProperties[1]
    height = videoProperties[2]
    fps = videoProperties[3]

    # The grayscale original video
    createVideoOutput(frames, width, height, fps, 'thema_1_2_originalGrayScaleVideo.avi')
    print('Original grayscale video exported successfully!')

    # Add all the frames to the original frames list
    originalFrames = [frames]
    H = entropyScore(originalFrames)
    print('Entropy of the original grayscale video is: ', H)

    # Calculate the motion vectors using the hierarchical search algorithm
    MVnSAD = []
    for i in range(1, len(frames)):
        print(f'Frame {i} of {len(frames)}')
        referenceFrame = frames[i - 1]
        targetFrame = frames[i]
        MVnSAD.append(hierarchicalSearch(referenceFrame, targetFrame, width, height))
    motionVectors = [[[mv] for mv, _ in value] for value in MVnSAD]

    ''' Temporary block of code to load the motion vectors from a file '''
    saveEncodedData(motionVectors, 'tempFile.pkl')
    # motionVectors = readEncodedData('tempFile.pkl')
    # motionVectors = [[[mv] for mv, _ in value] for value in motionVectors]
    ''' TO BE REMOVED '''

    # Calculate the motion compensated frames
    motionCompensatedFrames = motionCompensationForEncoding(frames, motionVectors)

    createVideoOutput(motionCompensatedFrames, width, height, fps, 'motion.avi')

    # Calculate the sequence error images
    seqErrorImages = calculateSeqErrorImages(frames, motionCompensatedFrames)
    createVideoOutput(seqErrorImages, width, height, fps, 'SeqError.avi')

    # ------------------------------------- Huffman encoding -------------------------------------#
    # Create the Huffman tree for the Motion Vectors
    huffmanTreeVectors = createHuffmanTreeVector(motionVectors)
    print('\tHuffman tree created for the vectors successfully!')

    # Create the Huffman table for the Motion Vectors
    huffmanCodeBookVectors = createHuffmanTableVector(huffmanTreeVectors)
    print('\tHuffman table created successfully!')

    # Encode the motion vectors for the Motion Vectors
    encodedMotionVectors = encodeHuffmanVector(motionVectors, huffmanCodeBookVectors)
    print('\tMotion vectors encoded successfully!')

    # Create the Huffman tree for the Sequence Error Images
    huffmanTreeSeqErrorImages = createHuffmanTree(seqErrorImages)
    print('\tHuffman tree created for the sequence error images successfully!')

    # Create the Huffman table for the Sequence Error Images
    huffmanCodeBookSeqErrorImages = createHuffmanTable(huffmanTreeSeqErrorImages)
    print('\tHuffman table created for the sequence error images successfully!')

    # Encode the sequence error images for the Sequence Error Images
    encodedSeqErrorImages = encodeHuffman(seqErrorImages, huffmanCodeBookSeqErrorImages)
    print('\tSequence error images encoded successfully!')

    # ------------------------------------------ Save data ---------------------------------------- #
    # Save the motion vectors
    saveEncodedData(encodedMotionVectors, 'thema_1_2_eMV.pkl')
    saveEncodedData(huffmanCodeBookVectors, 'thema_1_2_hCBV.pkl')
    print('\tMotion vectors saved successfully!')

    # Save the sequence error images
    saveEncodedData(encodedSeqErrorImages, 'thema_1_2_eSEI.pkl')
    saveEncodedData(huffmanCodeBookSeqErrorImages, 'thema_1_2_hCBSEI.pkl')

    # Save the video properties
    saveEncodedData(videoProperties, 'thema_1_2_vP.pkl')
    print('\tSequence error images saved successfully!')

    return 0


if __name__ == '__main__':
    entropy1 = videoEncoder()
