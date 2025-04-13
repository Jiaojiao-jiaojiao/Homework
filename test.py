import numpy as np
import pickle
import matplotlib.pyplot as plt
from model import ThreeLayerNN
from data_loader import load_cifar10_data
import os

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_weights(W1):
    """
    可视化第一层隐藏层的权重矩阵，每个神经元的权重向量将被重塑为 32x32x3 图像。
    """
    num_neurons = 16  # 要显示的神经元数量
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i in range(num_neurons):
        ax = axes[i // 4, i % 4]
        weight_vector = W1[:, i]  # 获取第i个神经元的权重向量
        
        # 将权重向量重塑为32x32x3图像
        weight_image = weight_vector.reshape(3, 32, 32).transpose(1, 2, 0)
        
        # 归一化到 [0, 1] 之间，防止颜色过暗或过亮
        weight_image = (weight_image - weight_image.min()) / (weight_image.max() - weight_image.min())
        
        ax.imshow(weight_image)
        ax.axis('off')
        ax.set_title(f'Neuron {i}')

    plt.suptitle('第一隐藏层神经元权重可视化', fontsize=16)
    plt.tight_layout()
    plt.savefig('weights_visualization.png')
    plt.show()


def test():
    # 加载 CIFAR-10 数据集
    X_train, y_train, X_test, y_test = load_cifar10_data('./cifar-10-batches-py')
    input_size = 3072
    hidden_size = 128  
    output_size = 10

    # 加载最优模型参数
    model_path = os.path.join(os.getcwd(), 'best_model.pkl')
    if not os.path.exists(model_path):
        print(f"错误: 找不到文件 {model_path}")
        return

    with open(model_path, 'rb') as f:
        best_params = pickle.load(f)

    # 创建并加载模型
    model = ThreeLayerNN(input_size, hidden_size, output_size)
    model.params = best_params

    # 预测并计算准确率
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"测试准确率: {acc:.4f}")

    # 可视化权重
    visualize_weights(model.params['W1'])

    # 加载训练曲线数据（loss & val_acc）
    curves_path = os.path.join(os.getcwd(), 'best_model_curves.pkl')
    if os.path.exists(curves_path):
        with open(curves_path, 'rb') as f:
            curves = pickle.load(f)
        loss_curve = curves['loss']
        val_acc_curve = curves['val_acc']
        epochs = np.arange(1, len(loss_curve) + 1)

        # 绘制训练过程中的损失和验证准确率曲线
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss_curve, marker='o')
        plt.title("训练集 Loss 曲线")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_acc_curve, marker='o', color='green')
        plt.title("验证集 Accuracy 曲线")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.tight_layout()
        plt.savefig("test_loss_valacc_curves.png")
        plt.show()
    else:
        print(f"提示: 找不到训练曲线文件 {curves_path}")


if __name__ == '__main__':
    test()



