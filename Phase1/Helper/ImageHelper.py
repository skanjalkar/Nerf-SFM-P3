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


