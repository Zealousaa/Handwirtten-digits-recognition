from model import M3
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tf
import torch.optim as opt
from torch.utils.data import DataLoader
from typing import *
import numpy as np
import matplotlib.pyplot as plt
import copy

# 是否在GPU上训练 ----------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

# 超参 --------------------------------------------------------
batch_size: int = 120
learning_rate: float = 1e-3
gamma: float = 0.98
epochs: int = 120

# 数据增强操作 ------------------------------------------------
transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.1307), (0.3081)),
    tf.RandomAffine(translate=(0.2, 0.2), degrees=0),
    tf.RandomRotation((-20, 20))
])

# 生成测试集和训练集的迭代器 -----------------------------------
train_sets = tv.datasets.MNIST("data", transform=transform)
test_sets = tv.datasets.MNIST("data", transform=tf.Compose([tf.ToTensor(),
                                                            tf.Normalize((0.1307), (0.3081))]),
                              train=False)

train_loader = DataLoader(
    train_sets,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10)  # num_workers 根据机子实际情况设置，一般<=CPU数

test_loader = DataLoader(
    test_sets,
    batch_size=batch_size,
    shuffle=True,
    num_workers=10)

data_loader = {"train": train_loader, "test": test_loader}
data_size = {"train": len(train_sets), "test": len(test_sets)}

# 导入模型 ---------------------------------------------------
model = M3().to(device)

# 损失函数, 与LogSoftmax配合 == CrossEntropyLoss -------------
criterion = nn.NLLLoss().to(device=device)

# Adam优化器, 初始学习率0.001, 其他参数全部默认就行-------------
optimizer = opt.Adam(model.parameters(), lr=learning_rate)

# 学习率指数衰减, 每轮迭代学习率更新为原来的gamma倍；verbose为True时，打印学习率更新信息
scheduler = opt.lr_scheduler.ExponentialLR(
    optimizer=optimizer, gamma=gamma, verbose=True)

# 训练 -------------------------------------------------------


def train_model(num_epochs: int = 1):

    best_acc: int = 0  # 记录测试集最高的预测得分, 用于辅助保存模型参数
    epochs_loss = np.array([])  # 记录损失变化，用于可视化模型
    epochs_crr = np.array([])  # 记录准确率变化，用于可视化模型

    for i in range(num_epochs):
        # 打印迭代信息
        print(f"epochs {i+1}/{num_epochs} :")

        # 每次迭代再分训练和测试两个部分
        for phase in ["train", "test"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss: float = 0.0
            running_crr: int = 0

            optimizer.zero_grad()  # 梯度清零

            for inputs, labels in data_loader[phase]:
                # 如果是在GPU上训练，数据需进行转换
                inputs, labels = inputs.to(device), labels.to(device)

                # 测试模型时不需要计算梯度
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)

                # 是否回溯更新梯度
                if phase != "test":
                    loss.backward()
                    optimizer.step()

                # 记录损失和准确预测数
                _, preds = torch.max(outputs, 1)
                running_crr += (preds == labels.data).sum()
                running_loss += loss.item() * inputs.size(0)

            # 打印结果和保存参数
            acc = (running_crr/data_size[phase]).cpu().numpy()
            if phase == "test" and acc > best_acc:
                best_acc = acc
                model_duplicate = copy.deepcopy(model)

            avg_loss = running_loss/data_size[phase]
            print(
                f"{phase} >>> acc:{acc:.4f},{running_crr}/{data_size[phase]}; loss:{avg_loss:.5f}")

            # 保存损失值和准确率
            epochs_crr = np.append(epochs_crr, acc)
            epochs_loss = np.append(epochs_loss, avg_loss)

        print(f"best test acc {best_acc:.4f}")
        scheduler.step()  # 更新学习率

    return model_duplicate, best_acc, epochs_crr, epochs_loss


def visualize_model(epochs_crr: np.array, epochs_loss: np.array):

    # 分离训练和测试损失与准确率
    epochs_crr = epochs_crr.reshape(-1, 2)
    epochs_loss = epochs_loss.reshape(-1, 2)
    train_crr, test_crr = epochs_crr[:, 0], epochs_crr[:, 1]
    train_lss, test_lss = epochs_loss[:, 0], epochs_loss[:, 1]

    # 绘制准确率变化图
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_crr)), train_crr, "-g", label="train")
    plt.plot(np.arange(len(test_crr)), test_crr, "-m", label="test")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.grid()

    # 绘制损失值变化图
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(train_lss)), train_lss, "-g", label="train")
    plt.plot(np.arange(len(test_lss)), test_lss, "-m", label="test")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()

    # 保存结果
    plt.tight_layout()
    plt.savefig(f"result.png")
    plt.close()


if __name__ == "__main__":
    trained_model, best_test_acc, crr, lss = train_model(epochs)
    visualize_model(crr, lss)
    torch.save(trained_model.state_dict(), f"model_params_{best_test_acc}.pth")
