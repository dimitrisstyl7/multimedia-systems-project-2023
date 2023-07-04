from huffman import *
from huffmanVectors import *
from motionCompensation import *
from videoFunction import *

videoPath = '../auxiliary2023/OriginalVideos/thema_1.avi'


def videoEncoder():
    """
        Encode the video
    """
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
    # motionVectors = hierarchicalSearch(frames, width, height)
    with open('../auxiliary2023/VideoProperties/motion_vectors.pkl', 'rb') as file:
        motionVectors = pickle.load(file)

    # Calculate the motion compensated frames
    motionCompensatedFrames = motionCompensationForEncoding(frames, motionVectors)

    # Huffman encoding
    # Create the Huffman tree for the Vectors
    motionVectors = np.array(motionVectors)
    huffmanTreeVectors = createHuffmanTreeVector(motionVectors)
    print('\tHuffman tree created for the vectors successfully!')

    # Create the Huffman table for the Vectors
    huffmanTableVectors = createHuffmanTableVector(huffmanTreeVectors)
    print('\tHuffman table created successfully!')

    # Encode the motion vectors for the Vectors
    encodedMotionVectors = encodeHuffmanVector(motionVectors, huffmanTableVectors)

    # Create the video of the error frames sequence
    seqErrorImages = calculateSeqErrorImages(frames, motionCompensatedFrames)

    # Huffman encoding
    # Create the Huffman tree for the seqErrorImages
    huffmanTreeError = createHuffmanTree(seqErrorImages)
    print('\tHuffman tree created for the seqErrorImages successfully!')

    # Create the Huffman table for the seqErrorImages
    huffmanTableError = createHuffmanTableVector(huffmanTreeError)
    print('\tHuffman table created for the seqErrorImages successfully!')

    # Encode the error frames sequence
    encodedMotionError = encodeHuffman(seqErrorImages, huffmanTableError)

    # createVideoOutput(seqErrorImages, width, height, fps, 'thema_1_2_seqErrorFrames.avi')

    videoSpecs = np.array([len(frames), width, height, fps], dtype='float64')
    saveEncodedVideo(encodedMotionVectors, 'thema_1_2_encodedMV.pkl', huffmanTableVectors, 'thema_1_2_hTV.pkl',
                     motionVectors[0].shape, 'thema_1_2_mVS.pkl')
    saveEncodedVideo(encodedMotionError, 'thema_1_2_encodedME.pkl', huffmanTableError, 'thema_1_2_hTE.pkl',
                     videoSpecs, 'thema_1_2_vS.pkl')
    print('\tEncoded video properties exported successfully!')
    return H  # Return the entropy of the original grayscale video


def videoDecoder():
    """
        Decode the video
    """
    encodedMotionVectors, huffmanTableVectors, motionVectorsSpecs = readVideoInfo('thema_1_2_encodedMV.pkl',
                                                                                  'thema_1_2_hTV.pkl',
                                                                                  'thema_1_2_mVS.pkl')
    encodedMotionError, huffmanTableError, videoSpecs = readVideoInfo('thema_1_2_encodedME.pkl',
                                                                      'thema_1_2_hTE.pkl', 'thema_1_2_vS.pkl')

    print('\tEncoded video properties imported successfully!')
    width = int(videoSpecs[1])
    height = int(videoSpecs[2])
    fps = float(videoSpecs[3])

    decodedMotionVectors = decodeHuffmanVector(encodedMotionVectors, huffmanTableVectors, motionVectorsSpecs[1],
                                               motionVectorsSpecs[0])

    decodedMotionError = decodeHuffman(encodedMotionError, huffmanTableError, width, height)

    i_frame = decodedMotionError[0]
    motion_compensated_frames = motionCompensationForDecoding(i_frame, decodedMotionVectors)
    return 1


if __name__ == '__main__':
    entropy1 = videoEncoder()
    entropy2 = videoDecoder()
    # if entropy1 == entropy2:
    #     print('The decoded video is the same as the original video!')
    # else:
    #     print('The decoded video is not the same as the original video! Something went wrong!')
