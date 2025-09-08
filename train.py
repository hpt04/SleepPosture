import torch
import torch.nn as nn
import torch.optim as optim
import os
from model.dataloader import create_dataloaders
from model.model import SleepPostureCNN

def train_model():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建数据加载器
    data_dir = "/home/jxchen/class/data/text_data"
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32
    )
    
    # 创建模型
    model = SleepPostureCNN(num_classes=4).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 100
    
    # 创建保存模型的目录
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device).view(-1)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device).view(-1)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100*val_correct/val_total
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {100*correct/total:.2f}%')
        print(f'Val Acc: {100*val_correct/val_total:.2f}%')
        print('-' * 50)
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_acc': val_acc,
        'loss': running_loss/len(train_loader)
    }, final_model_path)
    
    print(f'Training completed!')
    print(f'Final model saved to: {final_model_path}')

if __name__ == "__main__":
    train_model()