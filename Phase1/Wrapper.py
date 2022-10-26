import argparse
import numpy as np
from Helper.ImageHelper import *
from Helper.createPairMatches import *
from GetInliersRANSAC import *

def readCalibrationMatrix(path):
    '''Read the calibration matrix'''
    with open(path+"calibration.txt", 'r') as f:
        contents = f.read()

    K = []
    for i in contents.split():
        K.append(float(i))

    K = np.array(K)
    K = np.reshape(K, (3,3))
    return K


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data_path', default='../Data/MatchingData/', help='Data path')
    Parser.add_argument('--findImgPair', default=False, type=bool, help='To get the matches for all the pairs')
    Args = Parser.parse_args()
    data_path = Args.data_path
    findImgPair = Args.findImgPair

    if findImgPair:
        createMatchestxt(data_path)
    # get the camera calibration matrix
    K = readCalibrationMatrix(data_path)

    imgHelper = ImageHelper(data_path)
    images = imgHelper.readImages()
    # initial points(before ransac)
    matchPoints = imgHelper.readPoints("matches12.txt")

    # RANSAC
    ransacObj = RANSAC()
    ransacObj.getInliersRansac(matchPoints)



if __name__ == "__main__":
    main()