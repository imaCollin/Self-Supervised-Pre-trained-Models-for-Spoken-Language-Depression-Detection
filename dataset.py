import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_preprocessing import process_audio_file

class AudioDataset(Dataset):
    """
    自定义音频数据集类，用于加载音频文件及其标签。

    Args:
        file_paths (list): 音频文件路径列表。
        labels (list): 与音频对应的标签列表。
        feature_type (str): 要提取的特征类型（"mfcc", "mel", "wav2vec2"）。
        target_sr (int): 目标采样率，默认为16000。
        model_name (str): wav2vec2 模型名称（仅当 feature_type="wav2vec2" 时需要）。
    """
    def __init__(self, file_paths, labels, feature_type="mfcc", target_sr=16000, model_name=None):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_type = feature_type
        self.target_sr = target_sr
        self.model_name = model_name

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        返回指定索引的数据样本及其标签。

        Args:
            idx (int): 索引值。

        Returns:
            tuple: (特征, 标签)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 提取特征
        features = process_audio_file(
            file_path, feature_type=self.feature_type, target_sr=self.target_sr, model_name=self.model_name
        )

        # 转换为 Tensor
        features = torch.tensor(features, dtype=torch.float32)

        # 如果标签是标量，转换为 Tensor
        label = torch.tensor(label, dtype=torch.float32)

        return features, label

def split_dataset(file_paths, labels, train_ratio=0.8, val_ratio=0.1):
    """
    将数据集划分为训练集、验证集和测试集。

    Args:
        file_paths (list): 音频文件路径列表。
        labels (list): 音频标签列表。
        train_ratio (float): 训练集比例，默认0.8。
        val_ratio (float): 验证集比例，默认0.1。

    Returns:
        tuple: (训练集数据, 验证集数据, 测试集数据)
    """
    assert len(file_paths) == len(labels), "音频文件和标签数量不一致！"

    # 打乱数据
    indices = np.arange(len(file_paths))
    np.random.shuffle(indices)
    file_paths = np.array(file_paths)[indices]
    labels = np.array(labels)[indices]

    # 计算分割索引
    total_samples = len(file_paths)
    train_end = int(train_ratio * total_samples)
    val_end = train_end + int(val_ratio * total_samples)

    train_files = file_paths[:train_end]
    train_labels = labels[:train_end]

    val_files = file_paths[train_end:val_end]
    val_labels = labels[train_end:val_end]

    test_files = file_paths[val_end:]
    test_labels = labels[val_end:]

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)

def create_data_loaders(file_paths, labels, batch_size=32, feature_type="mfcc", target_sr=16000, model_name=None):
    """
    创建训练集、验证集和测试集的数据加载器。

    Args:
        file_paths (list): 音频文件路径列表。
        labels (list): 音频标签列表。
        batch_size (int): 批量大小，默认32。
        feature_type (str): 要提取的特征类型（"mfcc", "mel", "wav2vec2"）。
        target_sr (int): 目标采样率，默认为16000。
        model_name (str): wav2vec2 模型名称（仅当 feature_type="wav2vec2" 时需要）。

    Returns:
        tuple: (训练数据加载器, 验证数据加载器, 测试数据加载器)
    """
    # 数据分割
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = split_dataset(file_paths, labels)

    # 创建数据集
    train_dataset = AudioDataset(train_files, train_labels, feature_type, target_sr, model_name)
    val_dataset = AudioDataset(val_files, val_labels, feature_type, target_sr, model_name)
    test_dataset = AudioDataset(test_files, test_labels, feature_type, target_sr, model_name)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
