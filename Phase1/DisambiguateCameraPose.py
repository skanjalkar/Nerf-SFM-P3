import numpy as np
import matplotlib.pyplot as plt
import sys
from LinearTriangulation import *
from Helper.ImageHelper import *

class Disambiguate():
    def __init__(self, inliers, K) -> None:
        self.inliers = inliers
        self.K = K

    def chiralityCheck(self, X, R, C):
        count = 0
        r3 = R[2]
        for x in X:
            # pry()
            if(np.dot(r3, x-C).any())>0:
                count += 1
        return count

    def disambiguateCameraPose(self,C2, R2, P1,K):
        '''
        Inputs:
        C2
        R2
        P1
        K
        '''

        # find the maximum number of points in both
        # camera poses
        # the one with max points is the correct answer
        maxCount = 0
        plotHelper = Plot()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x1 = self.inliers[:, 0:2]
        x2 = self.inliers[:, 2:4]
        bestX, bestC, bestR = None, None, None
        # linTriangle = LinearTriangulation(self.K)
        index = 0
        for r2, c2 in zip(R2, C2):
            c2 = c2.reshape((3, 1))
            X2 = LinearTrinagulation(P1, c2, r2, K,x1, x2)
            # X2 = linTriangle.LinearTriangulation(C1, R1, c2, r2, x1, x2)
            # pry()
            count = self.chiralityCheck(X2, r2, c2)
            plotHelper.plotTriangle(X2, c2, r2, index)
            if count > maxCount:
                print(count)
                maxCount = count
                bestR = r2
                bestC = c2
                bestX = X2
                i = index
            index += 1
        # print("Found best x")

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        # plt.show()

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plotHelper.plotTriangle(bestX, bestC, bestR, i)
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        # # plt.show()

        return bestX, bestC, bestR, i
