import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=4,stride=1)

        self.fc1 = nn.Linear(in_features=120,out_features=84,bias=True)
        self.fc2 = nn.Linear(in_features=84,out_features=10,bias=True)

    def forward(self,x):

        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x