import numpy as np
import matplotlib.pyplot as plt
from LinearTriangulation import *
from Helper.ImageHelper import *

class Disambiguate():
    def __init__(self, inliers, K) -> None:
        self.maxCount = 0
        self.inliers = inliers
        self.K = K

    def disambiguateCameraPose(self, C1, R1, C2, R2):
        # find the maximum number of points in both
        # camera poses
        # the one with max points is the correct answer
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x1 = self.inliers[:, 0:2]
        x2 = self.inliers[:, 2:4]
        linTriangle = LinearTriangulation(self.K)
        for r2, c2 in zip(R2, C2):
            X2 = linTriangle.LinearTriangulation(C1, R1, c2, r2, x1, x2)
            count = 0
            for x in X2:
                if np.dot(r2[2], (x-c2))>0:
                    count += 1

            if count > self.maxCount:
                self.maxCount = count
                bestR = r2
                bestC = c2
                xBest = X2
        print("Found best x")

        plt.xlim(-15, 20)
        plt.ylim(-30, 40)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        imgHelper = Plot()
        
