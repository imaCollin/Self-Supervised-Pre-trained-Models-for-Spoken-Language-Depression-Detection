import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import create_data_loaders
from model import get_model
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    单个 epoch 的训练函数。

    Args:
        model (nn.Module): 待训练的模型。
        train_loader (DataLoader): 训练集数据加载器。
        criterion (nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。

    Returns:
        float: 当前 epoch 的平均训练损失。
    """
    model.train()
    running_loss = 0.0

    for features, labels in train_loader:
        # 将数据移动到设备
        features, labels = features.to(device), labels.to(device)

        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, labels.long())

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def validate(model, val_loader, criterion):
    """
    验证函数。

    Args:
        model (nn.Module): 待验证的模型。
        val_loader (DataLoader): 验证集数据加载器。
        criterion (nn.Module): 损失函数。

    Returns:
        tuple: (平均损失, 准确率)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for features, labels in val_loader:
            # 将数据移动到设备
            features, labels = features.to(device), labels.to(device)

            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels.long())
            running_loss += loss.item()

            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def test(model, test_loader):
    """
    测试函数。

    Args:
        model (nn.Module): 测试的模型。
        test_loader (DataLoader): 测试集数据加载器。

    Returns:
        float: 测试集准确率。
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for features, labels in test_loader:
            # 将数据移动到设备
            features, labels = features.to(device), labels.to(device)

            # 前向传播
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main():
    # 配置参数
    data_dir = "./data"  # 数据目录
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    model_name = "cnn_gru"  # 使用的模型名称 ("cnn_gru" 或 "simple_cnn")
    feature_type = "mfcc"  # 特征类型 ("mfcc", "mel", "wav2vec2")
    save_path = "./best_model.pth"

    # 加载数据集（需要准备好 file_paths 和 labels）
    file_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]  # 示例音频路径
    labels = [0, 1, 0]  # 示例标签

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        file_paths, labels, batch_size=batch_size, feature_type=feature_type
    )

    # 加载模型
    model = get_model(model_name=model_name, input_channels=1, num_classes=2).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)

        # 验证
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        # 打印日志
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")

    # 测试
    model.load_state_dict(torch.load(save_path))
    test_accuracy = test(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
