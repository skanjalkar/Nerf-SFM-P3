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

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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
    img1, img2 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    ransacObj = RANSAC()
    inlierPoints, outlierPoints, bestF, img1Pts, img2Pts = ransacObj.getInliersRansac(matchPoints)
    pts1, pts2 = [], []
    for i in range(len(inlierPoints)):
        img1x, img1y = int(inlierPoints[i][0]), int(inlierPoints[i][1])
        pts1.append((img1x, img1y))
        img2x, img2y = int(inlierPoints[i][2]), int(inlierPoints[i][3])
        pts2.append((img2x, img2y))
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,bestF)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,bestF)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    cv2.imshow('img3', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('img5', img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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