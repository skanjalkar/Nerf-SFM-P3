import os
import cv2
import pry
import numpy as np

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
            print(img1x, img1y)
            print()
            print(img2x, img2y)
            cv2.circle(appendedImage, (img1x, img1y), 3, (0, 0, 0), 1)
            cv2.circle(appendedImage, (img2x + np.int(image1.shape[1]), img2y), 3, (255, 0, 0), 1)
            cv2.line(appendedImage, (img1x, img1y), (img2x + np.int(image1.shape[1]), img2y), (0, 255, 0), 1)
        cv2.imshow("Inliers", appendedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plotOutliers(self, image1, image2, outliers):
        height = max(image1.shape[0], image2.shape[0])
        width = image1.shape[1]+image2.shape[1]
        appendedImage = np.zeros_like((height, width, 3))
        appendedImage[:image1.shape[0], :image1.shape[1], :] = image1
        appendedImage[:image2.shape[0], image1.shape[1]:, :] = image2

        for i in range(len(outliers)):
            img1x, img1y = outliers[i][0], outliers[i][1]
            img2x, img2y = outliers[i][2], outliers[i][3]

            cv2.circle(appendedImage, (img1x, img1y), 3, (0, 0, 0), 1)
            cv2.circle(appendedImage, (img2x + np.float32(image1.shape[1]), img2y), 3, (255, 0, 0), 1)
            cv2.line(appendedImage, (img1x, img1y), (img2x + np.float32(image1.shape[1]), img2y), (0, 255, 0), 1)
        cv2.imshow("Outliers", appendedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

