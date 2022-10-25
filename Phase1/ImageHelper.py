import os
import cv2

class ImageHelper():
    def __init__(self, path) -> None:
        self.path = path+"/Images/"

    def readImages(self):
        images = []
        for image in os.listdir(self.path):
            img = cv2.imread(os.path.join(self.path, image))
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            images.append(img)
        return images

    def readPoints(self):
        pass


