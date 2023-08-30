from hierarchicalSearch import hierarchicalSearch
from huffman import *
from huffmanVectors import *
from motionCompensation import *
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
    motionVectors = hierarchicalSearch(frames)

    # Calculate the motion compensated frames
    motionCompensatedFrames = motionCompensationForEncoding(frames, motionVectors, width, height)
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
    motionVectorsSpecs = (len(motionVectors[0]), len(motionVectors[0][0]))
    saveEncodedData(motionVectorsSpecs, 'thema_1_2_mVS.pkl')
    print('\tEncoded motion vectors saved successfully!')

    # Save the sequence error images
    saveEncodedData(encodedSeqErrorImages, 'thema_1_2_eSEI.pkl')
    saveEncodedData(huffmanCodeBookSeqErrorImages, 'thema_1_2_hCBSEI.pkl')
    print('\tEncoded sequence error images saved successfully!')

    # Save the video properties
    saveEncodedData(videoProperties, 'thema_1_2_vP.pkl')

    return H


def videoDecoder():
    """
        Decode the video
    """
    encodedMotionVectors = readEncodedData('thema_1_2_eMV.pkl')
    huffmanCodebookVectors = readEncodedData('thema_1_2_hCBV.pkl')
    motionVectorsSpecs = readEncodedData('thema_1_2_mVS.pkl')
    encodedMotionError = readEncodedData('thema_1_2_eSEI.pkl')
    huffmanCodeBookError = readEncodedData('thema_1_2_hCBSEI.pkl')
    videoProperties = readEncodedData('thema_1_2_vP.pkl')

    print('\tEncoded video properties imported successfully!')
    width = videoProperties[1]
    height = videoProperties[2]
    fps = videoProperties[3]

    # Decode the motion vectors
    decodedMotionVectors = decodeHuffmanVector(encodedMotionVectors, huffmanCodebookVectors, motionVectorsSpecs[1],
                                               motionVectorsSpecs[0])
    decodedMotionVectors = [
        [tuple(decodedMotionVectorsTuple.tolist()) for decodedMotionVectorsTuple in decodedMotionVectorsSubList] for
        decodedMotionVectorsSubList in decodedMotionVectors]

    # Decode the sequence of error frames
    decodedSeqErrorImages = decodeHuffman(encodedMotionError, huffmanCodeBookError, width, height)

    # Calculate the motion compensated frames
    motionCompensatedFrames = motionCompensationForDecoding(decodedMotionVectors, width, height, decodedSeqErrorImages)

    # Add error sequence to the motion compensated frames
    decodedFrames = addSeqErrorImagesToCompensatedFrames(motionCompensatedFrames, decodedSeqErrorImages)

    # Convert the list to a numpy array
    decodedFrames = np.array(decodedFrames, dtype='uint8')

    # Create the video of the decoded frames
    createVideoOutput(decodedFrames, width, height, fps, 'thema_1_2_decodedVideo.avi')
    print('Decoded grayscale video exported successfully!')

    H = entropyScore(decodedFrames)
    print('Entropy of the decoded grayscale video is: ', H)
    return H


if __name__ == '__main__':
    entropy1 = videoEncoder()
    entropy2 = videoDecoder()
