# Homework
# 🎯 三层神经网络图像分类器（CIFAR-10）

本项目基于 NumPy 实现了一个三层前馈神经网络，用于对 CIFAR-10 图像数据集进行分类。模型不依赖任何深度学习框架，适合学习和教学用途。

---

## 📌 项目简介

- **输入层**：3072维（32×32×3 彩色图像展平）
- **隐藏层1**：512个神经元，ReLU 激活函数
- **隐藏层2**：256个神经元，ReLU 激活函数
- **输出层**：10类，Softmax 输出概率分布
- **初始化方法**：Xavier 初始化
- **训练算法**：使用 Mini-batch 梯度下降 + 反向传播算法
- **目标任务**：图像分类（CIFAR-10 数据集）

---

## 📂 项目文件结构

```bash
.
├── model.py               # 神经网络结构实现
├── train.py               # 模型训练脚本
├── test.py                # 模型测试脚本
├── data_loader.py         # CIFAR-10 数据加载模块
├── best_model.pkl         # 训练过程中保存的最佳模型参数
├── best_model_curves.pkl  # 记录训练曲线数据（loss, acc）
├── cifar-10-batches-py/   # CIFAR-10 数据集（需提前解压）
└── README.md              # 项目说明文档
