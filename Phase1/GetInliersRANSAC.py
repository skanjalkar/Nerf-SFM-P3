import cv2
import numpy as np
from EstimateFundamentalMatrix import *
import pry

class RANSAC():
    def __init__(self) -> None:
        self.threshold = 0.05
        self.iterations=10000
        self.bestF = None

    def getInliersRansac(self, points):
        img1Pts = points[:, 0:2]
        img2Pts = points[:, 2:4]

        max_inliers = 0
        for i in range(self.iterations):

            randompts = np.random.choice(len(img1Pts), size=8)
            points1_8 = img1Pts[randompts, :]
            points2_8 = img2Pts[randompts, :]
            
            fObj = FMatrix(randompts)
            F = fObj.estimateFMatrix()

            x2TFx1 = np.abs(np.diag(np.dot(np.dot(img1Pts, F), img2Pts.T)))

            inliersNum = np.where(x2TFx1<self.threshold)
            outliersNum = np.where(x2TFx1>=self.threshold)

            if np.shape(inliersNum[0])[0]>max_inliers:
                max_inliers = np.shape(inliersNum[0])[0]
                maxNumInliers = inliersNum
                maxNumOutliers = outliersNum
                self.bestF = F



