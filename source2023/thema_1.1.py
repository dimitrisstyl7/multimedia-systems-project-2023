from source2023.imageFunction import *
from source2023.videoFunction import *

videoPath = "../auxiliary2023/OriginalVideos/thema_1.avi"


def videoEncoder():
    """
    Encode the original uncompressed video using Huffman encoding.
    """
    # Read the video
    frames, fps = openVideo(videoPath)
    print(
        f'The video has {len(frames)} frames, a height of {frames[0].shape[0]} pixels, a width of {frames[0].shape[1]} pixels and a framerate of {fps} frames per second.')

    # Convert the video to grayscale
    frames = createGrayscaleVideo(frames)

    # # The grayscale original video
    # createVideoOutput(frames, fps, 'thema_1_1_originalGrayScaleVideo.avi')

    originalFrames = []
    seqErrorImages = []

    # Add all the frames to the original frames list
    originalFrames.append(frames)

    H = entropy_score(frames)

    print("Entropy of the original grayscale video is: ", H)

    # Add the first frame to the error frames list (I frame)
    seqErrorImages.append(frames[0])

    # Add the rest of the frames to the error frames list (P(n+1) - P(n) -> frame)
    for P in range(1, len(frames)):
        # Calculate the error image of the current frame
        errorImage = calculateErrorImage(frames[P], frames[P - 1])

        # Add the error image to the error frames list
        seqErrorImages.append(convertToUint8(errorImage))

    # Convert the error frames list to a numpy array
    seqErrorImages = np.array(seqErrorImages, dtype='uint8')

    # Save the video properties to a binary file
    videoSpecs = np.array([len(frames), frames[0].shape[0], frames[0].shape[1], fps], dtype='float64')

    print(f'The video has {len(seqErrorImages)} error frames.')
    frame = seqErrorImages[131]
    huffmanCompression(frame, frame.size)


def huffmanCompression(frame, N):
    # Create a dictionary with the symbols and their probabilities
    symbols = np.unique(frame, return_counts=True)  # Find the unique symbols and their counts

    # Transpose the array to have the symbols in the first column and their counts in the second
    tempSymbols = np.array(symbols).T.tolist()

    # Create a # list of probabilities where the index of each probability corresponds to each symbol
    probList = [x[1] / N for x in tempSymbols]

    probList = [0.4, 0.3, 0.15, 0.1, 0.05]
    # probList = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    # Create the Huffman codebook
    codebook = huffmanEncode(probList.copy())
    print(codebook)


def huffmanEncode(probList):
    # Huffman encoding
    def findMin():
        nonlocal foundInStartingProbList_1, foundInStartingProbList_2, whoAmI  # nonlocal is used to access variables
        # from the outer scope
        minProb = min(probList)
        minIdx = probList.index(minProb)
        if minIdx < size:
            probList[minIdx] = 2  # Set the minimum probability to 2 so that it is not found again
            if whoAmI == 1:
                foundInStartingProbList_1 = True
            else:  # whoAmI == 2
                foundInStartingProbList_2 = True

        else:
            probList.pop(minIdx)
        return minProb, minIdx

    size = len(probList)  # The number of symbols
    encodingList = ['' for i in range(len(probList))]
    sumDict = {}

    # min1 and min2 are the two smallest probabilities (min1 < min2)
    # min1 -> 0
    # min2 -> 1

    while probList[-1] != 1:
        # if len(probList) == size + 1 and probList.count(2) == size:
        #     break  # End of the algorithm

        # Find the two smallest probabilities
        foundInStartingProbList_1 = False
        foundInStartingProbList_2 = False
        whoAmI = 1
        minProb_1, minIdx_1 = findMin()
        whoAmI = 2
        minProb_2, minIdx_2 = findMin()

        # Fill the encoding list
        prob_1_found = False
        prob_2_found = False
        symbolsOfProb_1 = None
        symbolsOfProb_2 = None

        for symbols, prob in sumDict.items():
            if (foundInStartingProbList_1 and foundInStartingProbList_2) or prob_1_found and prob_2_found:
                break
            if not foundInStartingProbList_1 and not prob_1_found and minProb_1 == prob:
                prob_1_found = True
                symbolsOfProb_1 = symbols  # GH
                continue
            elif not foundInStartingProbList_2 and not prob_2_found and minProb_2 == prob:
                prob_2_found = True
                symbolsOfProb_2 = symbols
                continue

        if prob_1_found and prob_2_found:
            sumDict.pop(symbolsOfProb_1)
            sumDict.pop(symbolsOfProb_2)
            sumDict[symbolsOfProb_1 + symbolsOfProb_2] = minProb_1 + minProb_2
            probList.append(minProb_1 + minProb_2)
            for symbol in symbolsOfProb_1:
                encodingList[int(symbol)] += '0'
            for symbol in symbolsOfProb_2:
                encodingList[int(symbol)] += '1'
        elif prob_1_found:
            sumDict.pop(symbolsOfProb_1)
            sumDict[symbolsOfProb_1 + str(minIdx_2)] = minProb_1 + minProb_2
            probList.append(minProb_1 + minProb_2)
            for symbol in symbolsOfProb_1:
                encodingList[int(symbol)] += '0'
            encodingList[minIdx_2] += '1'
        elif prob_2_found:
            sumDict.pop(symbolsOfProb_2)
            sumDict[symbolsOfProb_2 + str(minIdx_1)] = minProb_1 + minProb_2
            probList.append(minProb_1 + minProb_2)
            encodingList[minIdx_1] += '0'
            for symbol in symbolsOfProb_2:
                encodingList[int(symbol)] += '1'
        else:
            sumDict[str(minIdx_1) + str(minIdx_2)] = minProb_1 + minProb_2
            probList.append(minProb_1 + minProb_2)
            encodingList[minIdx_1] += '0'
            encodingList[minIdx_2] += '1'
    return encodingList

    # # Save the encoded error frames to a binary file
    # with open('../auxiliary2023/EncodedVideos/thema_1_1_encodedErrorFrames.pkl', 'wb') as file:
    #     pickle.dump(encodedErrorFrames, file)

    # # Calculate the entropy of the error frames sequence
    # H = entropy_score(seqErrorImages)
    # print("Entropy of the seqErrorFrames (grayscale) video is: ", H)
    #
    # # Create the video of the error frames sequence
    # createVideoOutput(seqErrorImages, fps, 'thema_1_1_seqErrorFrames.avi')
    #
    # # To help the decoder we will save the video properties
    # saveVideoInfo(seqErrorImages, 'thema_1_1_seqErrorFrames.pkl', videoSpecs, 'thema_1_1_videoSpecs.pkl')


def videoDecoder():
    """
    Decode the encoded video using the dictionary of the Huffman.
    """
    framesNumber, videoSpecs = readVideoInfo('thema_1_1_seqErrorFrames.pkl', 'thema_1_1_videoSpecs.pkl')

    num_frames = int(videoSpecs[0])
    height = int(videoSpecs[1])
    width = int(videoSpecs[2])
    loaded_fps = float(videoSpecs[3])

    print(
        f'The video has {num_frames} frames, a height of {height} pixels, a width of {width} pixels and a framerate of {loaded_fps} frames per second.')

    frames = np.reshape(framesNumber, (num_frames, height, width))

    # Create the output video
    createVideoOutput(frames, loaded_fps, 'thema_1_1_decodedVideo.avi')

    first_frame_flag = True

    decodedFrames = []

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

    decodedFrames = np.array(decodedFrames, dtype='uint8')

    createVideoOutput(decodedFrames, videoSpecs[3], 'thema_1_1_decodedVideo.avi')


if __name__ == '__main__':
    videoEncoder()
    videoDecoder()
