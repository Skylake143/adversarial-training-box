import torch
import torch.nn as nn
import torch.nn.functional as F

class EMNIST_NET_256x2(nn.Module):
    def __init__(self):
        super(EMNIST_NET_256x2, self).__init__()
        self.name = "emnist_net_256x2"
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 47) # TODO: eventually adjustable

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x