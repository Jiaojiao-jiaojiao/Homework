import numpy as np
from model import ThreeLayerNN
from data_loader import load_cifar10_data
import matplotlib.pyplot as plt
import pickle
import itertools
import os
import seaborn as sns
import pandas as pd


# 训练一个三层神经网络模型，并返回验证集准确率、最优参数、副产物loss记录和验证集准确率记录
def train_one_model(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, learning_rate, reg_lambda, num_epochs=20, batch_size=100):
    model = ThreeLayerNN(input_size, hidden_size, output_size, reg_lambda=reg_lambda)
    best_val_acc = 0
    best_params = None
    loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[permutation], y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            Z1, A1, Z2, A2 = model.forward(X_batch)
            loss = model.compute_loss(A2, y_batch)
            model.backward(X_batch, y_batch, Z1, A1, A2, learning_rate)

        if epoch % 10 == 0:
            learning_rate *= 0.5

        y_val_pred = model.predict(X_val)
        val_acc = np.mean(y_val_pred == y_val)
        loss_history.append(loss)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = model.params.copy()

    return best_val_acc, best_params, loss_history, val_acc_history


def grid_search_train():
    X_train, y_train, X_test, y_test = load_cifar10_data('./cifar-10-batches-py')

    val_size = 5000
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    input_size = 3072
    output_size = 10

    learning_rates = [1e-1, 1e-2]
    hidden_sizes = [64, 128, 256]
    reg_lambdas = [1e-2, 1e-3]

    best_val_acc = 0
    best_hyperparams = None
    best_curves = None
    results = []

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    print("\n===== 开始网格搜索 =====")
    for lr, hs, reg in itertools.product(learning_rates, hidden_sizes, reg_lambdas):
        print(f"\n正在训练: 学习率={lr}, 隐藏层大小={hs}, 正则项={reg}")
        val_acc, best_params, loss_history, val_acc_history = train_one_model(
            X_train, y_train, X_val, y_val,
            input_size, hs, output_size,
            learning_rate=lr,
            reg_lambda=reg
        )

        results.append({
            'learning_rate': lr,
            'hidden_size': hs,
            'reg_lambda': reg,
            'val_accuracy': val_acc
        })

        model_name = f"model_lr{lr}_h{hs}_reg{reg}.pkl"
        with open(os.path.join('saved_models', model_name), 'wb') as f:
            pickle.dump(best_params, f)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_hyperparams = (lr, hs, reg)
            best_curves = {'loss': loss_history, 'val_acc': val_acc_history}
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(best_params, f)
            with open('best_model_curves.pkl', 'wb') as f:
                pickle.dump(best_curves, f)

    print("\n===== 网格搜索完成，结果如下 =====")
    for r in results:
        print(f"学习率={r['learning_rate']}, 隐藏层={r['hidden_size']}, 正则={r['reg_lambda']} --> 验证集准确率={r['val_accuracy']:.4f}")
    print(f"\n最优组合：学习率={best_hyperparams[0]}, 隐藏层={best_hyperparams[1]}, 正则={best_hyperparams[2]} (验证集准确率={best_val_acc:.4f})")

    # 绘制热力图
    df = pd.DataFrame(results)
    pivot = df.pivot_table(index='hidden_size', columns='learning_rate', values='val_accuracy', aggfunc='max')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap='YlGnBu')
    plt.title("验证集准确率热力图 (固定正则项)")
    plt.xlabel("学习率")
    plt.ylabel("隐藏层大小")
    plt.tight_layout()
    plt.savefig('grid_search_heatmap.png')
    plt.show()



if __name__ == '__main__':
    grid_search_train()
