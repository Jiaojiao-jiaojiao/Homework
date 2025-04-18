# Homework
# NumPy三层神经网络图像分类器（CIFAR-10）

本项目基于 NumPy 实现了一个三层前馈神经网络，从头实现前向传播、反向传播、交叉熵损失与训练过程，完成对 CIFAR-10 图像分类任务。

---

##  项目简介

- **输入层**：3072维（32×32×3 彩色图像展平）
- **隐藏层**：支持不同大小（64、128、256），ReLU 激活函数
- **输出层**：10类，Softmax 输出概率分布
- **初始化方法**：Xavier 初始化
- **训练算法**：使用梯度下降 + 反向传播算法
- **目标任务**：图像分类（CIFAR-10 数据集）

---

## 项目文件结构
```bash
.
├── model.py               # 神经网络结构实现
├── data_loader.py         # CIFAR-10 数据加载模块
├── train.py               # 模型训练脚本
├── test.py                # 模型测试脚本
├── best_model.pkl         # 训练过程中保存的最佳模型参数
├── best_model_curves.pkl  # 记录训练曲线数据（loss, acc）
├── cifar-10-batches-py/   # CIFAR-10 数据集（需提前解压）
├── image                  # 项目相关图像
└── README.md              # 项目说明文档
```
---

##  模型结构

- 输入层：3072（32×32×3）
- 隐藏层 ：支持不同大小（64、128、256）+ ReLU 
- 输出层：10 类别 + Softmax

---

## 超参数搜索

项目支持对以下超参数组合进行网格搜索，自动保存最优模型：

- 学习率（learning rate）：`1e-1`, `1e-2`
- 隐藏层大小（hidden size）：`64`, `128`, `256`
- 正则化强度（reg lambda）：`1e-2`, `1e-3`

共计训练组合：2 × 3 × 2 = 12 种配置。

训练过程将自动输出验证集准确率。

---

## 训练过程

训练过程包含以下主要步骤：

### 前向传播

在每一次训练迭代中，输入图像数据通过网络的每一层进行前向传播，最终计算出输出层的预测结果。具体过程如下：

- 输入层：将原始的32x32的RGB图像展平成3072维的向量。

- 隐藏层：每个隐藏层将计算加权和并应用ReLU激活函数，输出下一个层的输入。

- 输出层：经过最后一个隐藏层后，输出层通过Softmax激活函数生成一个10维的概率分布，表示每个类别的预测概率。

---

### 损失计算

网络的损失函数为交叉熵损失函数，该损失函数计算预测类别与实际标签之间的差异，用于优化网络权重。公式如下：

$$ L = - \sum_{i=1}^{C} y_i \log(\hat{y}_i) $$

其中，$y_i$ 是实际类别标签，$\hat{y}_i$ 是模型预测的类别概率，$C$ 是类别数（本项目中为10）。

---

### 反向传播

通过反向传播算法计算梯度，更新每层的权重和偏置。反向传播分为以下步骤：

- 计算输出层的误差，基于损失函数的梯度；

- 将误差通过网络反向传播，计算每个隐藏层的误差；

- 利用链式法则计算每层权重和偏置的梯度；

- 使用梯度下降法更新权重和偏置。

---

### 模型优化

在每次迭代中，使用梯度下降法（SGD）来优化网络参数。更新公式为：

$$ W^{[l]} = W^{[l]} - \eta \frac{\partial J}{\partial W^{[l]}} $$

其中，$\eta$ 为学习率，$\frac{\partial J}{\partial W^{[l]}}$ 为当前层权重的梯度。

---

### 训练过程记录

训练过程中记录了每一轮的训练损失和验证集准确率，以便评估训练过程和模型收敛情况。

- 每一轮训练后，保存当前模型的权重和训练曲线数据（包括损失和准确率）。

- 每完成一个超参数组合的训练，输出当前最佳模型的验证集准确率。

## 测试集评估

最终模型在测试集上的表现：

- 准确率：51.66%

通过在测试集上的评估，展示了训练过程中选择的超参数配置在实际数据上的泛化能力。

---

## 总结

- 实现了完整的前向 + 反向传播流程
- 训练过程可视化，便于调参与观察收敛
- 权重可视化展示了网络学习的感知能力

### 后续工作

- 使用优化器如 Adam 替代 SGD
- 加入 Dropout / BatchNorm 提升泛化能力
- 尝试更深层的网络结构（4 层以上）

---

## 模型参数下载

- 百度网盘下载地址：
  https://pan.baidu.com/s/11cxM8u-TrnMr752iCajm-w  
  提取码：`yykj`

---






