import cv2
import numpy as np
from EstimateFundamentalMatrix import *

class RANSAC():
    def __init__(self) -> None:
        self.threshold = 0.05
        self.bestF = None

    def getInliersRansac(self, points):
        max_inliers = 0
        for i in range(10000):
            randompts = np.random.choice(points, 8)
            randompts = np.array(randompts, np.float32)
            fObj = FMatrix(randompts)
            F = fObj.estimateFMatrix()

            x2TFx1 = np.abs(np.diag(np.dot(np.dot(points_image2, F), points_image1.T)))

            inliersNum = np.where(x2TFx1<self.threshold)
            outliersNum = np.where(x2TFx1>=self.threshold)

            if np.shape(inliersNum[0])[0]>max_inliers:
                max_inliers = np.shape(inliersNum[0])[0]
                maxNumInliers = inliersNum
                maxNumOutliers = outliersNum
                self.bestF = F

        

