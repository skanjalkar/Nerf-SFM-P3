import argparse
import numpy as np
from ImageHelper import *
from createPairMatches import *

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data_path', default='../Data/MatchingData/', help='Data path')
    Parser.add_argument('--findImgPair', default=False, type=bool, help='To get the matches for all the pairs')
    Args = Parser.parse_args()
    data_path = Args.data_path
    findImgPair = Args.findImgPair

    if findImgPair:
        createMatchestxt(data_path)

    # -------------------------------------------------------------------------#
    with open(data_path+"calibration.txt", 'r') as f:
        contents = f.read()

    K = []
    for i in contents.split():
        K.append(float(i))

    K = np.array(K)
    K = np.reshape(K, (3,3))
    # -------------------------------------------------------------------------#
    imgHelper = ImageHelper(data_path)
    images = imgHelper.readImages()



if __name__ == "__main__":
    main()