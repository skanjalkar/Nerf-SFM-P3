import torch
import torch.nn as nn
from dataParser import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # testing data loading
    data = Data("lego-20221031T225340Z-001/lego/")
    trainImages = data.readImages('train')
    trainRotation, trainTransformation = data.readJson('transforms_train.json')
    
    # testImages = data.readImages('test')
    # testRotation, testTransformation = data.readJson('transforms_train.json')


if __name__ == "__main__":
    main()
