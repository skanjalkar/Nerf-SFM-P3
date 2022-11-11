import cv2
import numpy as np
from EstimateFundamentalMatrix import *
import random
import sys
import math

class RANSAC():
    def __init__(self) -> None:
        self.threshold = 0.01
        self.iterations=10000
        self.bestF = None
        self.maxNumInliers = 0
        self.maxNumOutliers = math.inf

    def getInliersRansac(self, points):
        fObj = FMatrix()
        img1pts = points[:, 0:2]
        img2pts = points[:, 2:4]
        #convert imgpts to homogenous, otherwise
        # you get a dimenstional error when trying to calculate x2TFx1
        img1Pts = np.hstack((img1pts, np.ones((img1pts.shape[0], 1))))
        img2Pts = np.hstack((img2pts, np.ones((img2pts.shape[0], 1))))
        random.seed(100)
        # for consistent results, helps in debugging
        max_inliers = 0
        for i in range(self.iterations):
            randompts = np.random.choice(len(img1Pts), size=8)
            # randompts = np.array(random.sample(list(points), 8), np.float32)
            # pry()
            points1_8 = img1Pts[randompts, :]
            points2_8 = img2Pts[randompts, :]
            # pry()
            pts_to_F_matrix = []
            for i in range(len(points1_8)):
                pts_to_F_matrix.append((points1_8[i][0], points1_8[i][1], points2_8[i][0], points2_8[i][1]))


            F = fObj.estimateFMatrix(pts_to_F_matrix)
            # F = fObj.estimateFMatrix(randompts)
            x2TFx1 = np.abs(np.diag(np.dot(np.dot(img2Pts, F), img1Pts.T)))
            inliersNum = np.where(x2TFx1<self.threshold)
            outliersNum = np.where(x2TFx1>=self.threshold)

            if np.shape(inliersNum[0])[0]>max_inliers:
                max_inliers = np.shape(inliersNum[0])[0]
                self.maxNumInliers = inliersNum
                self.maxNumOutliers = outliersNum
        if self.maxNumInliers == 0:
            print(f'No inliers found for the given threshold, consider increasing the threshold and run the code again')
            print()
            print(f'------------------------------------------------------------------------')
            sys.exit()
        self.bestF = fObj.estimateFMatrix(points[self.maxNumInliers])
        print(self.bestF)
        print('-0-----------------------------------------------------')
        # pry()
        # F, mask = cv2.findFundamentalMat(points[:,:2], points[:, 2:4], cv2.FM_LMEDS)
        # print(self.maxNumInliers)
        # print(F)
        return points[self.maxNumInliers], points[self.maxNumOutliers], self.bestF, img1pts, img2pts
        # ptsleft = points[:,:2][mask.ravel() == 1]
        # ptsright = points[:,2:4][mask.ravel() == 1]
        # pry()
        # return ptsleft, ptsright, F, img1pts, img2pts
