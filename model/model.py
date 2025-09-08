import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepPostureCNN(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, height=40, width=26):
        super(SleepPostureCNN, self).__init__()
        
        self.height = height
        self.width = width
        self.input_channels = input_channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # (1, 40, 26) -> (32, 40, 26)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)              # (32, 40, 26) -> (64, 40, 26)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)             # (64, 40, 26) -> (128, 40, 26)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)  # 每次池化后尺寸减半
        
        # 计算全连接层输入维度
        # 经过3次池化: 40/8=5, 26/8=3.25->3, 所以最终特征图大小为 5x3
        self.fc_input_size = 128 * 5 * 3  # 128 * 5 * 3 = 1920
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout和BatchNorm
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # 输入: (batch, channels, height, width) = (batch, 1, 40, 26)
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor (batch, channels, height, width), got shape: {x.shape}")
        
        # 第一个卷积块: Conv -> BN -> ReLU -> Pool
        x = self.conv1(x)        # (batch, 32, 40, 26)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)         # (batch, 32, 20, 13)
        
        # 第二个卷积块: Conv -> BN -> ReLU -> Pool
        x = self.conv2(x)        # (batch, 64, 20, 13)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)         # (batch, 64, 10, 6)
        
        # 第三个卷积块: Conv -> BN -> ReLU -> Pool
        x = self.conv3(x)        # (batch, 128, 10, 6)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)         # (batch, 128, 5, 3)
        
        # 展平特征图
        x = x.view(x.size(0), -1)  # (batch, 128*5*3) = (batch, 1920)
        
        # 全连接层
        x = F.relu(self.fc1(x))  # (batch, 512)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # (batch, 128)
        x = self.dropout(x)
        x = self.fc3(x)          # (batch, 4)
        
        return x