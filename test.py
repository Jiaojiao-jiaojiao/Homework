import numpy as np
import pickle
import matplotlib.pyplot as plt
from model import ThreeLayerNN
from data_loader import load_cifar10_data
import os


def test():
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

    model = ThreeLayerNN(input_size, hidden_size, output_size)
    model.params = best_params

    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"测试准确率: {acc:.4f}")

    # 加载训练曲线数据（loss & val_acc）
    curves_path = os.path.join(os.getcwd(), 'best_model_curves.pkl')
    if os.path.exists(curves_path):
        with open(curves_path, 'rb') as f:
            curves = pickle.load(f)
        loss_curve = curves['loss']
        val_acc_curve = curves['val_acc']
        epochs = np.arange(1, len(loss_curve) + 1)

        # 绘图
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
        plt.rcParams['axes.unicode_minus'] = False
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



