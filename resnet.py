import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
            stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
            stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)  # <0 归0  ； >0保留
        return out


class ResNet(nn.Module):
    def __init__(self, hidden_sizes, num_blocks, input_dim,
        in_channels, n_classes):
        super(ResNet, self).__init__()
        assert len(num_blocks) == len(hidden_sizes) #断言 为True正常执行，为False引发异常
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.test = self.in_channels
        self.conv1 = nn.Conv1d(3, self.in_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        
        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1) # [1, 2, 2, ... ]
        # emumerate 返回对象（遍历序号，每个数据）
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],stride=strides[idx])) # 每一层的层定义
        self.encoder = nn.Sequential(*layers) # Sequential容器 按照构造函数中传递的顺序 添加到模块中
        self.z_dim = self._get_encoding_size() # 获得 encoder 输出 维度
        self.linear = nn.Linear(self.z_dim, self.n_classes) # 执行一个线性变换


    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x) # 内涵多个残差模块
        z = x.view(x.size(0), -1) # 把 X 拉成 1维的
        return z

    def forward(self, x):
        z = self.encode(x)
        a = self.linear(z)
        return a


    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self):
        temp = Variable(torch.rand(64, 3, self.input_dim)) # Variable 变量
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim


def add_activation(activation='relu'):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'elu':
        return nn.ELU(alpha=1.0)
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leaky relu':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    # SOFTPLUS DOESN'T WORK with automatic differentiation in pytorch
    elif activation == 'softplus':
        return nn.Softplus(beta=1, threshold=20)