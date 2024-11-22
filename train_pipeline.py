import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_data_loaders
from model import get_model
from sklearn.metrics import accuracy_score
from pathlib import Path
import json

# ===============================
# 配置参数
# ===============================

class Config:
    DATASET_DIR = "./data/DAIC-WOZ"  # 数据集目录，适配 DAIC-WOZ 或 EATD
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    MODEL_NAME = "cnn_gru"  # 支持 "cnn_gru" 或 "simple_cnn"
    FEATURE_TYPE = "mfcc"  # 支持 "mfcc", "mel", 或 "wav2vec2"
    TARGET_SR = 16000  # 目标采样率
    MODEL_SAVE_PATH = "./best_model.pth"  # 模型保存路径
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# 数据加载函数
# ===============================

def load_daic_woz_data(dataset_dir):
    """
    加载 DAIC-WOZ 数据集的音频文件路径和标签。

    Args:
        dataset_dir (str): 数据集目录路径。

    Returns:
        tuple: (file_paths, labels)
    """
    file_paths = []
    labels = []

    # 假设音频文件和元数据位于子目录中
    for subdir in Path(dataset_dir).iterdir():
        if not subdir.is_dir():
            continue

        # 加载元数据（假设有 `transcript.json` 或类似文件）
        metadata_file = subdir / "transcript.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                label = metadata.get("PHQ8_binary")  # 获取二分类标签（抑郁或非抑郁）

                # 获取音频文件路径（假设主音频文件名为 `audio.wav`）
                audio_file = subdir / "audio.wav"
                if audio_file.exists():
                    file_paths.append(str(audio_file))
                    labels.append(label)

    return file_paths, labels


# ===============================
# 训练函数
# ===============================

def train_model():
    # 加载 DAIC-WOZ 数据集
    print(f"Loading dataset from {Config.DATASET_DIR}...")
    file_paths, labels = load_daic_woz_data(Config.DATASET_DIR)

    # 检查数据是否加载成功
    if not file_paths or not labels:
        raise ValueError(f"No data found in {Config.DATASET_DIR}!")

    print(f"Loaded {len(file_paths)} samples from dataset.")

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        file_paths, labels,
        batch_size=Config.BATCH_SIZE,
        feature_type=Config.FEATURE_TYPE,
        target_sr=Config.TARGET_SR
    )

    # 加载模型
    print(f"Initializing model: {Config.MODEL_NAME}...")
    model = get_model(model_name=Config.MODEL_NAME, input_channels=1, num_classes=2).to(Config.DEVICE)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 训练和验证
    best_val_accuracy = 0.0
    for epoch in range(Config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)

            # 前向传播
            outputs = model(features)
            loss = criterion(outputs, labels.long())

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)

                outputs = model(features)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"Best model saved to {Config.MODEL_SAVE_PATH}")

    # 测试阶段
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(Config.DEVICE), labels.to(Config.DEVICE)

            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    train_model()
