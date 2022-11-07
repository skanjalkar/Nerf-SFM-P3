import torch
import numpy as np
import torch.nn as nn
from dataParser import *
import matplotlib.pyplot as plt
import pry
from train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def main():
    # testing data loading
    data = Data("lego-20221031T225340Z-001/lego/")
    trainImages = data.readImages('train')
    trainRotation, trainTransformation = data.readJson('transforms_train.json')

    focalLength = np.array([138.8888789])
    focalLength = torch.from_numpy(focalLength).to(device)

    height, width = trainImages.shape[1:3]
    near_threshold = 2.
    far_threshold = 6.

    print(trainImages.shape)
    # pry()

    trainTransformation = torch.from_numpy(trainTransformation).to(device)

    trainImages = torch.from_numpy(trainImages[:, ..., :3]).to(device)

    trainObj = Train(height, width, trainImages, trainTransformation, focalLength, near_threshold, far_threshold)
    trainObj.train(6, 32, 16384)
    # ro, rd = trainObj.get_ray_bundle(trainTransformation[0])
    # query_points, depth_values = trainObj.computeQueryPoints(ro, rd, 32)


    # testImages = data.readImages('test')
    # testRotation, testTransformation = data.readJson('transforms_train.json')


if __name__ == "__main__":
    main()
