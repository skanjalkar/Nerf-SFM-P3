import cv2
import os
import json
import numpy as np

class Data():
    def __init__(self, path) -> None:
        self.path = path


    def readImages(self, type):
        images = []
        image_order = sorted(os.listdir(self.path+f'{type}/'))
        for image in image_order:
            img = cv2.imread(os.path.join(self.path+f'{type}/', image))
            images.append(img)
        return np.array(images)

    def readJson(self, type):
        f = open(self.path+type)

        data = json.load(f)
        transformationMatrix = []
        rotation = []
        for frame in data['frames']:
            rotation.append(float(frame["rotation"]))
            transform_matrix = frame["transform_matrix"]
            numpyTransform = np.asarray(transform_matrix, dtype=np.float32)
            transformationMatrix.append(numpyTransform)

        f.close()
        return np.array(rotation), np.array(transformationMatrix)

