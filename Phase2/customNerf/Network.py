import torch

class NerfModel(torch.nn.Module):
    def __init__(self, filter_size=128, encoding=6) -> None:
        super(NerfModel, self).__init__()
        self.layer1 = torch.nn.Linear(3 + 3*2*encoding, filter_size)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        self.relu = torch.nn.functional.relu
        self.float()

    def forward(self, x):
        x = x.float()
        # print("BEFORE LAYER 1", x.shape, x.dtype)
        x = self.relu(self.layer1(x))
        # print("BEFORE LAYER 2", x.shape, x.dtype)
        x = self.relu(self.layer2(x))
        # print("BEFORE LAYER 3", x.shape, x.dtype)
        x = self.layer3(x)
        # print("FINAL SHAPE", x.shape, x.dtype)
        return x
