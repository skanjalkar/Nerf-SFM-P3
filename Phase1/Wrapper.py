import argparse
import numpy as np
from Helper.ImageHelper import *
from Helper.createPairMatches import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamanentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *

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

    P1 = np.dot(K, np.hstack((np.identity(3), np.zeros((3,1)))))
    R1, C1 = P1[:3, :3], P1[:, -1]
    # pry()
    imgHelper = ImageHelper(data_path)
    images = imgHelper.readImages()
    # initial points(before ransac)
    matchPoints = imgHelper.readPoints("matches12.txt")
    matchPoints = np.array(matchPoints, np.float32)

    # RANSAC
    ransacObj = RANSAC()
    inlierPoints, outlierPoints, bestF, img1Pts, img2Pts = ransacObj.getInliersRansac(matchPoints)
    imgHelper.plotInliers(images[0], images[1], inlierPoints, "Inliers", False)
    imgHelper.plotOutliers(images[0], images[1], outlierPoints)

    # Essential matrix
    eObj = EMatrix(bestF, K)
    E = eObj.getEssentialMatrix()

    # Camera Pose
    cameraPoseObj = CameraPose(E)
    potentailC2, potentailR2 = cameraPoseObj.cameraPoses()

    # LinearTriangulation
    disObj = Disambiguate(inlierPoints, K)
    bestX, bestC, bestR, index = disObj.disambiguateCameraPose(C1, R1, potentailC2, potentailR2, P1)




if __name__ == "__main__":
    main()