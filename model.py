import torch
import torch.nn as nn

class SimpleAudioCNN(nn.Module):
    """
    简单的 CNN 模型，仅包含卷积层和全连接层。

    Args:
        input_channels (int): 输入通道数。
        cnn_output_channels (int): CNN 卷积层的输出通道数。
        num_classes (int): 输出类别数，默认为2（二分类）。
    """
    def __init__(self, input_channels=1, cnn_output_channels=32, num_classes=2):
        super(SimpleAudioCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn_output_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(cnn_output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(cnn_output_channels, cnn_output_channels * 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(cnn_output_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_output_channels * 2 * 16 * 16, 128),  # 假设输入特征大小为 (1, 64, 64)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征，形状为 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 模型输出，形状为 (batch_size, num_classes)。
        """
        # CNN 提取特征
        x = self.cnn(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)
        return x

class AudioDepressionModel(nn.Module):
    """
    音频抑郁检测模型，结合 CNN 和 GRU。

    Args:
        input_channels (int): 输入通道数。
        cnn_output_channels (int): CNN 卷积层的输出通道数。
        gru_hidden_size (int): GRU 的隐藏层大小。
        num_classes (int): 输出类别数，默认为2（二分类）。
    """
    def __init__(self, input_channels=1, cnn_output_channels=32, gru_hidden_size=64, num_classes=2):
        super(AudioDepressionModel, self).__init__()

        # 卷积神经网络 (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn_output_channels, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(cnn_output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(cnn_output_channels, cnn_output_channels * 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(cnn_output_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # GRU 层
        self.gru = nn.GRU(input_size=cnn_output_channels * 2, hidden_size=gru_hidden_size, num_layers=2,
                          bidirectional=True, batch_first=True)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入特征，形状为 (batch_size, channels, height, width)。

        Returns:
            torch.Tensor: 模型输出，形状为 (batch_size, num_classes)。
        """
        # CNN 提取局部特征
        x = self.cnn(x)  # 输出形状: (batch_size, cnn_output_channels * 2, height, width)

        # 调整维度以适配 GRU 输入
        x = x.permute(0, 2, 1, 3)  # 调整为 (batch_size, height, channels, width)
        batch_size, time_steps, channels, features = x.shape
        x = x.reshape(batch_size, time_steps, -1)  # 调整为 (batch_size, time_steps, features)

        # GRU 处理序列特征
        x, _ = self.gru(x)  # 输出形状: (batch_size, time_steps, gru_hidden_size * 2)
        x = x[:, -1, :]  # 取最后一个时间步的输出

        # 全连接层
        x = self.fc(x)  # 输出形状: (batch_size, num_classes)
        return x

def get_model(model_name="cnn_gru", input_channels=1, num_classes=2):
    """
    根据名称获取模型实例。

    Args:
        model_name (str): 模型名称，支持 "cnn_gru" 和 "simple_cnn"。
        input_channels (int): 输入通道数。
        num_classes (int): 输出类别数，默认为2（二分类）。

    Returns:
        nn.Module: PyTorch 模型实例。
    """
    if model_name == "cnn_gru":
        return AudioDepressionModel(input_channels=input_channels, num_classes=num_classes)
    elif model_name == "simple_cnn":
        return SimpleAudioCNN(input_channels=input_channels, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
