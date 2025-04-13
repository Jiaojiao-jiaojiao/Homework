import pickle
import numpy as np
import os


def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        return X.reshape(-1, 3, 32, 32).astype("float"), np.array(Y)


def load_cifar10_data(data_dir):
    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(data_dir, f'data_batch_{b}')
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    return X_train, y_train, X_test, y_test