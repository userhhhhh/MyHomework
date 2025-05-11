from datasets import load_dataset
import os
from PIL import Image

# 加载 MNIST-M 数据集
dataset = load_dataset("Mike0307/MNIST-M")

# 创建目录结构
os.makedirs("mnist_m/mnist_m_train", exist_ok=True)
os.makedirs("mnist_m/mnist_m_test", exist_ok=True)

# 保存训练集
with open("mnist_m/mnist_m_train_labels.txt", "w") as f:
    for idx, example in enumerate(dataset['train']):
        image = example['image']
        label = example['label']
        filename = f"{idx:06d}.png"
        image.save(f"mnist_m/mnist_m_train/{filename}")
        f.write(f"{filename} {label}\n")

# 保存测试集
with open("mnist_m/mnist_m_test_labels.txt", "w") as f:
    for idx, example in enumerate(dataset['test']):
        image = example['image']
        label = example['label']
        filename = f"{idx:06d}.png"
        image.save(f"mnist_m/mnist_m_test/{filename}")
        f.write(f"{filename} {label}\n")

