import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, num_points=2048):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 对每个点的三个坐标应用一个全连接层
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, num_points * 3)  # 输出层的大小为点数乘以每个点的坐标数

        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入x的形状为(batch_size, num_points, 3)
        batch_size = x.size(0)
        x = x.view(-1, 3)  # 展平为(batch_size * num_points, 3)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        x = x.view(batch_size, -1, 3)  # 重塑为(batch_size, num_points, 3)
        return x
