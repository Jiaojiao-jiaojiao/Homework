import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', reg_lambda=0.0):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * 0.01,
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * 0.01,
            'b2': np.zeros((1, output_size))
        }
        self.activation = activation
        self.reg_lambda = reg_lambda

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        Z1 = X.dot(W1) + b1
        A1 = self.relu(Z1) if self.activation == 'relu' else self.sigmoid(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def compute_loss(self, A2, Y):
        m = Y.shape[0]
        log_probs = -np.log(A2[range(m), Y])
        data_loss = np.sum(log_probs) / m
        reg_loss = self.reg_lambda / 2 * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))
        return data_loss + reg_loss

    def backward(self, X, Y, Z1, A1, A2, learning_rate):
        m = X.shape[0]
        W1, W2 = self.params['W1'], self.params['W2']

        dZ2 = A2
        dZ2[range(m), Y] -= 1
        dZ2 /= m

        dW2 = A1.T.dot(dZ2) + self.reg_lambda * W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(W2.T)
        dZ1 = dA1 * (self.relu_derivative(Z1) if self.activation == 'relu' else self.sigmoid_derivative(Z1))

        dW1 = X.T.dot(dZ1) + self.reg_lambda * W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.params['W1'] -= learning_rate * dW1
        self.params['b1'] -= learning_rate * db1
        self.params['W2'] -= learning_rate * dW2
        self.params['b2'] -= learning_rate * db2

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=1)