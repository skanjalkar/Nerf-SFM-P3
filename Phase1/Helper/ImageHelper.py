import os
import cv2
import pry
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import math


class ImageHelper():
    def __init__(self, path) -> None:
        self.ImagePath = path+"/Images/"
        self.path = path

    def readImages(self):
        images = []
        image_order = sorted(os.listdir(self.ImagePath))
        for image in image_order:
            # print(image)
            img = cv2.imread(os.path.join(self.ImagePath, image))
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            images.append(img)
        return images

    def readPoints(self, file_name):
        file = self.path + file_name
        with open(file, 'r') as f:
            lines = f.readlines()

        pts = []
        for line in lines:
            uc, vc, ui, vi, r, g, b = line.split()
            pts.append([np.float32(uc), np.float32(vc), np.float32(ui), np.float32(vi), np.float32(r), np.float32(g), np.float32(b)])

        pts = np.array(pts)
        return pts

    def plotInliers(self, image1, image2, inliers, matchesBW, save=False):
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1]+image2.shape[1]
        # pry()
        appendedImage = np.zeros((height, width, 3), type(image1.flat[0]))
        appendedImage[:image1.shape[0], :image1.shape[1], :] = image1
        appendedImage[:image2.shape[0], image2.shape[1]:, :] = image2
        # pry()
        # cv2.imshow("sidebyside", appendedImage)
        # cv2.waitKey(0)

        for i in range(len(inliers)):
            img1x, img1y = int(inliers[i][0]), int(inliers[i][1])
            img2x, img2y = int(inliers[i][2]), int(inliers[i][3])

            cv2.circle(appendedImage, (img1x, img1y), 3, (0, 0, 0), 1)
            cv2.circle(appendedImage, (img2x + np.int(image1.shape[1]), img2y), 3, (255, 0, 0), 1)
            cv2.line(appendedImage, (img1x, img1y), (img2x + np.int(image1.shape[1]), img2y), (0, 255, 0), 1)
        cv2.imshow("Inliers", appendedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plotOutliers(self, image1, image2, outliers):
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1]+image2.shape[1]

        appendedImage = np.zeros((height, width, 3), type(image1.flat[0]))
        appendedImage[:image1.shape[0], :image1.shape[1], :] = image1
        appendedImage[:image2.shape[0], image1.shape[1]:, :] = image2

        for i in range(len(outliers)):
            img1x, img1y = int(outliers[i][0]), int(outliers[i][1])
            img2x, img2y = int(outliers[i][2]), int(outliers[i][3])

            cv2.circle(appendedImage, (img1x, img1y), 3, (0, 0, 0), 1)
            cv2.circle(appendedImage, (img2x + np.int(image1.shape[1]), img2y), 3, (255, 0, 0), 1)
            cv2.line(appendedImage, (img1x, img1y), (img2x + np.int(image1.shape[1]), img2y), (0, 0, 255), 1)
        cv2.imshow("Outliers", appendedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Plot():
    def __init__(self) -> None:
        pass


    def plotTriangle(self, bestX, bestC, bestR, i):
        bestX = np.array(bestX)
        # pry()
        # bestX = np.reshape((bestX.shape[0], 3))

        colors = np.array(['y', 'b', 'c', 'r'])
        ax = plt.gca()
        ax.plot(0, 0, marker=mpl.markers.CARETDOWN, markersize=15, color='k')

        eulerAngles = self.rotationMatrixToEulerAngles(bestR)
        cameraAngle = np.rad2deg(eulerAngles)

        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        # pry()
        t._transform = t.get_transform().rotate_deg(int(cameraAngle[1]))

        x = bestX[:, 0]
        z = bestX[:, 2]

        ax.scatter((bestC[0]), (bestC[1]), marker=t, s=250, color=colors[i])
        ax.scatter(x, z, s=4, color=colors[i])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')



    # https://learnopencv.com/rotation-matrix-to-euler-angles/
    def isRotationMatrix(self, R):
        Rt = R.T
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])