import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(4096, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), 2, 2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(F.max_pool2d(self.bn4(self.conv4(x)), 2, 2))
        x = F.relu(F.max_pool2d(self.bn5(self.conv5(x)), 2, 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    