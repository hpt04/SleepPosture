import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path

class SleepPostureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        睡姿数据集
        Args:
            data_dir: 数据目录路径 (如 /home/jxchen/class/data/text_data)
            transform: 数据变换函数
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # 睡姿标签映射 - 4类分类（移除坐姿）
        self.posture_labels = {
            '1': 0,  # 仰卧
            '2': 1,  # 俯卧
            '3': 2,  # 左侧卧（弯腿）-> 左侧卧
            '4': 2,  # 左侧卧（伸腿）-> 左侧卧
            '5': 3,  # 右侧卧（弯腿）-> 右侧卧
            '6': 3   # 右侧卧（伸腿）-> 右侧卧
        }
        
        self.frames = []  # 存储所有帧数据
        self.labels = []  # 存储对应的标签
        
        # 遍历所有用户文件夹
        for user_folder in self.data_dir.iterdir():
            if user_folder.is_dir() and user_folder.name != 'sit':
                # 获取文件夹中的所有txt文件并排序
                txt_files = sorted(list(user_folder.glob('*.txt')))
                
                # 检查是否有29个文件，如果是则使用新的标签分配方式
                if len(txt_files) == 29:
                    for i, txt_file in enumerate(txt_files):
                        # 先检查文件是否有效（有足够的数据）
                        try:
                            file_data = self._load_txt_file(txt_file)
                            if file_data.size == 0:  # 跳过空文件
                                continue
                        except Exception:
                            continue  # 跳过无法读取的文件
                            
                        # 确定标签
                        if i < 6:  # 前6个文件：仰卧
                            label = 0
                        elif i < 15:  # 第7-15个文件：俯卧
                            label = 1
                        elif i < 22:  # 第16-22个文件：左侧卧
                            label = 2
                        else:  # 第23-29个文件：右侧卧
                            label = 3
                            
                        # 将每一帧作为单独的样本
                        for frame in file_data:
                            self.frames.append(frame)
                            self.labels.append(label)
                # 检查是否有21个文件，按照文件名结尾数字分配标签
                elif len(txt_files) == 21:
                    for txt_file in txt_files:
                        # 先检查文件是否有效（有足够的数据）
                        try:
                            file_data = self._load_txt_file(txt_file)
                            if file_data.size == 0:  # 跳过空文件
                                continue
                        except Exception:
                            continue  # 跳过无法读取的文件
                            
                        # 从文件名提取结尾数字
                        file_num = int(txt_file.stem.split('_')[-1])
                        
                        # 确定标签
                        if 1 <= file_num <= 6:  # 1-6结尾：仰卧
                            label = 0
                        elif 7 <= file_num <= 9:  # 7-9结尾：俯卧
                            label = 1
                        elif 10 <= file_num <= 15:  # 10-15结尾：左侧卧
                            label = 2
                        elif 16 <= file_num <= 21:  # 16-21结尾：右侧卧
                            label = 3
                            
                        # 将每一帧作为单独的样本
                        for frame in file_data:
                            self.frames.append(frame)
                            self.labels.append(label)
                else:
                    # 原有的基于文件名的标签映射（适用于7个文件的情况）
                    for txt_file in txt_files:
                        # 从文件名提取标签 (如 czy_1.txt -> 标签1)
                        label_num = txt_file.stem.split('_')[-1]
                        # 跳过文件名以_7结尾的文件（坐姿数据）
                        if label_num == '7':
                            continue
                        if label_num in self.posture_labels:
                            # 先检查文件是否有效（有足够的数据）
                            try:
                                file_data = self._load_txt_file(txt_file)
                                if file_data.size == 0:  # 跳过空文件
                                    continue
                            except Exception:
                                continue  # 跳过无法读取的文件
                                
                            label = self.posture_labels[label_num]
                            # 将每一帧作为单独的样本
                            for frame in file_data:
                                self.frames.append(frame)
                                self.labels.append(label)
        
        # 不再处理sit文件夹（已移除坐姿分类）
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        # 获取单帧数据
        frame = self.frames[idx]  # shape: (40, 26)
        label = self.labels[idx]
        
        # 添加通道维度，使其变为 (1, 40, 26)
        data = np.expand_dims(frame, axis=0)
        
        if self.transform:
            data = self.transform(data)
        
        return torch.FloatTensor(data), torch.LongTensor([label])
    
    def _load_txt_file(self, file_path):
        """加载txt文件数据"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                # 解析每行数据（40x26的帧）
                # 处理逗号分隔和空格分隔两种格式
                line_data = line.strip()
                if ',' in line_data:
                    values = [float(x) for x in line_data.split(',')]
                else:
                    values = [float(x) for x in line_data.split()]
                if len(values) == 26:  # 确保是完整的一行
                    data.append(values)
        
        # 转换为numpy数组并reshape为帧格式
        data = np.array(data)
        if len(data) < 40:  # 检查是否有足够的行数
            # 返回空数组，让上层处理
            return np.array([])
            
        # 重塑为 (frames, height, width) 格式
        frames = len(data) // 40
        data = data[:frames*40].reshape(frames, 40, 26)
        
        return data
    


# 数据预处理函数
class DataTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize
    
    def __call__(self, data):
        # 检查数据是否为空
        if data.size == 0:
            raise ValueError(f"Empty data array encountered. Check if the input file has valid data.")
            
        if self.normalize:
            # 归一化到[0,1]范围
            data = data / 255.0 if data.max() > 1 else data
        
        return data

# 创建数据加载器的函数
def create_dataloaders(data_dir, batch_size=32, train_split=0.8):
    """
    创建训练和验证数据加载器
    """
    # 创建数据变换
    transform = DataTransform(normalize=True)
    
    # 创建完整数据集
    full_dataset = SleepPostureDataset(
        data_dir=data_dir,
        transform=transform
    )
    
    # 划分训练和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

# 使用示例
if __name__ == "__main__":
    # 创建数据加载器
    data_dir = "/home/jxchen/class/data/text_data"
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32
    )
    
    # 测试数据加载
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels.flatten()}")
        if batch_idx == 0:
            break