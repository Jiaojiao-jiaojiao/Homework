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

##  项目文件结构

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

---

## 模型结构

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

### 验证集准确率热力图

> 深色表示准确率较高的超参数组合

![验证集准确率热力图](https://raw.githubusercontent.com/Jiaojiao-jiaojiao/Homework/main/image/Figure_1.png)
在正则化参数不变的条件下，准确率较高的一组是：学习率为1e-1，隐藏层大小为256。

---

## 训练过程可视化

训练过程中记录了：

- 每一轮的训练损失
- 验证集准确率变化

### 损失下降曲线 + 验证准确率趋势

![训练损失曲线 + 验证准确率曲线](https://raw.githubusercontent.com/Jiaojiao-jiaojiao/Homework/main/image/Figure_2.png)

> 损失函数下降明显，验证集准确率上升趋稳，模型逐渐收敛，显示模型泛化良好。

---

## 权重可视化（隐藏层）

我们将第一隐藏层中部分神经元的权重向量 `(3072,)` 重新 reshape 成 `(3, 32, 32)` 并显示为 RGB 图像。

![隐藏层权重可视化](https://raw.githubusercontent.com/Jiaojiao-jiaojiao/Homework/main/image/Figure_3.png)

> 神经元权重图展示了不同感受野：有的偏向边缘，有的偏向颜色块，说明模型具备基础图像感知能力。

---

## 🧪 测试集评估

最终模型在测试集上的表现：

- **准确率：51.66%**

### 测试结果可视化图表

![测试集评估图](https://raw.githubusercontent.com/Jiaojiao-jiaojiao/Homework/main/image/Figure_4.png)

---

## 总结

- ✅ 实现了完整的前向 + 反向传播流程
- ✅ 训练过程可视化，便于调参与观察收敛
- ✅ 权重可视化展示了网络学习的感知能力

### 后续工作

- 使用优化器如 Adam 替代 SGD
- 加入 Dropout / BatchNorm 提升泛化能力
- 尝试更深层的网络结构（4 层以上）

---

## 模型参数下载

- 📥 百度网盘下载地址：
  https://pan.baidu.com/s/11cxM8u-TrnMr752iCajm-w  
  提取码：`yykj`

---






