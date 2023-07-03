from huffman import *
from imageFunction import *
from videoFunction import *

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"


def videoEncoder():
    """
        Encode the video
    """
    # Read the video
    frames, video_properties = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {video_properties[2]} pixels, a width of'
        f' {video_properties[1]} pixels and a framerate of {video_properties[3]} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    width = video_properties[1]
    height = video_properties[2]
    fps = video_properties[3]

    # The grayscale original video
    createVideoOutput(frames, width, height, fps, 'thema_1_1_originalGrayScaleVideo.avi')
    print("Original grayscale video exported successfully!")

    # Add all the frames to the original frames list
    originalFrames = [frames]

    seqErrorImages = calculateSeqErrorImages(frames)

    videoSpecs = np.array([len(frames), width, height, fps], dtype='float64')

    H = entropy_score(originalFrames)
    print("Entropy of the original grayscale video is: ", H)

    # Create the video of the error frames sequence
    createVideoOutput(seqErrorImages, width, height, fps, 'thema_1_1_seqErrorFrames.avi')

    # Huffman encoding
    # Create the Huffman tree
    huffmanTree = createHuffmanTree(seqErrorImages)
    print("\tHuffman tree created successfully!")
    # Create the Huffman table
    huffmanTable = createHuffmanTable(huffmanTree)
    print("\tHuffman table created successfully!")
    # Encode the error frames sequence
    encodedSeqErrorImages = encodeHuffman(seqErrorImages, huffmanTable)
    # Save the encoded error frames sequence
    saveEncodedVideo(encodedSeqErrorImages, 'thema_1_1_encodedSF.pkl', huffmanTable, 'thema_1_1_hT.pkl', videoSpecs,
                     'thema_1_1_vS.pkl')
    print("\tEncoded video properties exported successfully!")

    return H


def videoDecoder():
    """
        Decode the video
    """

    encodedSeqErrorImages, huffmanTable, videoSpecs = readVideoInfo('thema_1_1_encodedSF.pkl', 'thema_1_1_hT.pkl',
                                                                    'thema_1_1_vS.pkl')
    print("\tEncoded video properties imported successfully!")
    width = int(videoSpecs[1])
    height = int(videoSpecs[2])
    fps = float(videoSpecs[3])

    # Huffman decoding
    # Decode the error frames sequence
    frames = decodeHuffman(encodedSeqErrorImages, huffmanTable, width, height)

    print(
        f'The video has {len(frames)} frames, a height of {height} pixels, a width of {width} pixels and a framerate of'
        f' {fps} frames per second.')

    first_frame_flag = True

    decodedFrames = []

    # Recreate the frames of the original video
    prev_frame = None
    i = 0
    for frame in frames:
        if first_frame_flag:
            decodedFrames.append(frame)
            saveImage(frame, str(i) + '.jpg')
            prev_frame = frame
            first_frame_flag = False
        else:
            decodedFrame = prev_frame + frame  # Add frame to prev_frame
            saveImage(decodedFrame, str(i) + '.jpg')
            decodedFrames.append(decodedFrame)
            prev_frame = decodedFrame
        i += 1

    # Convert the list to a numpy array
    decodedFrames = np.array(decodedFrames, dtype='uint8')

    # Create the video of the decoded frames
    createVideoOutput(decodedFrames, width, height, fps, 'thema_1_1_decodedVideo.avi')
    print("Decoded grayscale video exported successfully!")

    H = entropy_score(decodedFrames)
    print("Entropy of the decoded grayscale video is: ", H)

    return H


if __name__ == '__main__':
    entropy1 = videoEncoder()
    entropy2 = videoDecoder()
    if entropy1 == entropy2:
        print("The decoded video is the same as the original video!")
    else:
        print("The decoded video is not the same as the original video! Something went wrong!")
