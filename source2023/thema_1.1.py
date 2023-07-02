from source2023.videoFunction import *
from source2023.imageFunction import *
from source2023.huffman import *
import numpy as np

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"


def videoEncoder():
    """
    Encode the video
    """
    # Read the video
    frames, video_properties = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {video_properties[2]} pixels, a width of {video_properties[1]} pixels and a framerate of {video_properties[3]} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    width = video_properties[1]
    height = video_properties[2]
    fps = video_properties[3]

    # The grayscale original video
    createVideoOutput(frames, width, height, fps, 'thema_1_1_originalGrayScaleVideo.avi')

    originalFrames = []
    seqErrorImages = []

    # Add all the frames to the original frames list
    originalFrames.append(frames)

    # Add the first frame to the error frames list (I frame)
    seqErrorImages.append(frames[0])

    # Create the Encoding Differential Pulse Code Modulation - DPCM
    for P in range(1, len(frames)):
        # Calculate the error image of the current frame
        errorImage = calculateErrorImage(frames[P], frames[P - 1])

        # Add the error image to the error frames list
        seqErrorImages.append(errorImage)

    seqErrorImages = np.array(seqErrorImages, dtype='uint8')

    print(seqErrorImages)

    videoSpecs = np.array([len(frames), width, height, fps], dtype='float64')

    print("Entropy of the original grayscale video is: ", entropy_score(originalFrames))

    # Calculate the entropy of the error frames sequence
    H = entropy_score(seqErrorImages)
    print("Entropy of the seqErrorFrames (grayscale) video is: ", H)

    # Create the video of the error frames sequence
    createVideoOutput(seqErrorImages, width, height, fps, 'thema_1_1_seqErrorFrames.avi')

    # Huffman encoding
    # Create the Huffman tree
    huffmanTree = createHuffmanTree(seqErrorImages)

    # Create the Huffman table
    huffmanTable = createHuffmanTable(huffmanTree)

    # Encode the error frames sequence
    encodedSeqErrorImages = encodeHuffman(seqErrorImages, huffmanTable)

    # Save the encoded error frames sequence
    saveEncodedVideo(encodedSeqErrorImages, 'thema_1_1_encodedSF.pkl', huffmanTable, 'thema_1_1_hT.pkl', videoSpecs,
                     'thema_1_1_vS.pkl')
    print("Encoded video saved successfully!")


def videoDecoder():
    """
    Decode the video
    """

    encodedSeqErrorImages, huffmanTable, videoSpecs = readVideoInfo('thema_1_1_encodedSF.pkl', 'thema_1_1_hT.pkl',
                                                                    'thema_1_1_vS.pkl')

    width = int(videoSpecs[1])
    height = int(videoSpecs[2])
    fps = float(videoSpecs[3])

    # Huffman decoding
    # Decode the error frames sequence
    frames = decodeHuffman(encodedSeqErrorImages, huffmanTable, width, height)

    print(f'frames decoded!')

    print(frames)
    print(
        f'The video has {len(frames)} frames, a height of {height} pixels, a width of {width} pixels and a framerate of {fps} frames per second.')

    #decodedFrames = reconstructFrames(frames, frames[0])
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


if __name__ == '__main__':
    videoEncoder()
    videoDecoder()
