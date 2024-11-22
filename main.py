from config import config
from dataset import AudioDataset
from model import DepressionDetectionModel
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    # 加载数据
    train_loader, val_loader = load_data(config["data_path"], config["batch_size"])

    # 初始化模型
    model = DepressionDetectionModel()
    model.to(device)

    # 设置损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, config["num_epochs"])

    # 验证模型
    accuracy, auc = evaluate_model(model, val_loader)
    print(f"Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
