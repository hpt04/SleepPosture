import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os


# 设备设置
def get_device():
    """安全地获取设备"""
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.init()
            return torch.device('npu'), 'NPU'
        elif torch.cuda.is_available():
            return torch.device('cuda'), 'CUDA'
        else:
            return torch.device('cpu'), 'CPU'
    except:
        return torch.device('cpu'), 'CPU'


device, device_name = get_device()
print(f"Using device: {device_name}")


# 定义与训练时相同的模型架构
class SleepPostureCNN(nn.Module):
    def __init__(self, num_classes=4, input_channels=1, height=40, width=26):
        super(SleepPostureCNN, self).__init__()

        self.height = height
        self.width = width
        self.input_channels = input_channels

        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 计算全连接层输入维度
        self.fc_input_size = 128 * 5 * 3

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

        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# 加载模型函数
def load_model(model_path, device):
    """加载训练好的模型"""
    # 创建模型实例
    model = SleepPostureCNN(num_classes=4)

    # 检查模型文件是否存在
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 将模型移动到相应设备
    model = model.to(device)
    model.eval()  # 设置为评估模式

    print(f"Model loaded from: {model_path}")
    print(f"Training epochs: {checkpoint.get('epoch', 'N/A')}")
    print(f"Final validation accuracy: {checkpoint.get('final_val_acc', 'N/A'):.2f}%")

    return model


# 数据预处理函数（与训练时保持一致）
def preprocess_data(data):
    """预处理输入数据，与训练时保持一致"""
    # 归一化到[0,1]范围
    data = data / 255.0 if data.max() > 1 else data
    return data


# 加载单帧数据
def load_single_frame(file_path):
    """从文本文件加载单帧数据"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # 解析每行数据
            line_data = line.strip()
            if ',' in line_data:
                values = [float(x) for x in line_data.split(',')]
            else:
                values = [float(x) for x in line_data.split()]
            if len(values) == 26:  # 确保是完整的一行
                data.append(values)

    # 转换为numpy数组
    data = np.array(data)

    # 检查数据形状
    if data.shape[0] != 40:
        raise ValueError(f"Expected 40 rows, got {data.shape[0]}")

    return data


# 使用模型进行预测
def predict(model, input_data, device):
    """使用模型进行预测"""
    # 预处理数据
    input_data = preprocess_data(input_data)

    # 添加批次和通道维度
    if input_data.ndim == 2:  # (height, width)
        input_data = np.expand_dims(input_data, axis=0)  # (1, height, width)
        input_data = np.expand_dims(input_data, axis=0)  # (1, 1, height, width)

    # 转换为张量
    input_tensor = torch.FloatTensor(input_data).to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class, probabilities.cpu().numpy()


# 主函数
def main():
    # 模型路径
    model_path = "saved_models/final_model.pth"

    try:
        # 加载模型
        model = load_model(model_path, device)

        # 睡姿类别映射
        posture_classes = {
            0: "仰卧 (Supine)",
            1: "俯卧 (Prone)",
            2: "左侧卧 (Left Side)",
            3: "右侧卧 (Right Side)"
        }

        # 示例：创建一些测试数据
        print("\n示例预测:")

        # 方法1: 使用随机生成的测试数据
        test_data = np.random.rand(40, 26)  # 随机生成40x26的数据
        predicted_class, probabilities = predict(model, test_data, device)

        print(f"随机测试数据:")
        print(f"  预测睡姿: {posture_classes[predicted_class]}")
        print(f"  各类别概率: {[f'{p:.4f}' for p in probabilities[0]]}")

        # 方法2: 从文件加载数据
        # 假设您有一些测试文件
        test_files_dir = "/path/to/your/test/files"  # 修改为您的测试文件路径
        if Path(test_files_dir).exists():
            print(f"\n从文件加载测试数据:")
            for test_file in Path(test_files_dir).glob("*.txt"):
                try:
                    frame_data = load_single_frame(test_file)
                    predicted_class, probabilities = predict(model, frame_data, device)

                    print(f"文件: {test_file.name}")
                    print(f"  预测睡姿: {posture_classes[predicted_class]}")
                    print(f"  各类别概率: {[f'{p:.4f}' for p in probabilities[0]]}")
                except Exception as e:
                    print(f"处理文件 {test_file.name} 时出错: {e}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行训练脚本 train.py 来训练模型")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    main()