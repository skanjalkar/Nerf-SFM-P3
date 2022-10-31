import os
import cv2
import numpy as np

class ImageHelper():
    def __init__(self, path) -> None:
        self.ImagePath = path+"/Images/"
        self.path = path

    def readImages(self):
        images = []
        for image in os.listdir(self.ImagePath):
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

        appendedImage = np.zeros_like((height, width, 3))
        appendedImage[:image1.shape[0], :image1.shape[1], :] = image1
        appendedImage[:image2.shape[0], image1.shape[1]:, :] = image2

        for i in range(len(inliers)):
            img1x, img1y = inliers[i][0], inliers[i][1]
            img2x, img2y = inliers[i][2], inliers[i][3]

            cv2.circle(appendedImage, (img1x, img1y), 3, (0, 0, 0), 1)


