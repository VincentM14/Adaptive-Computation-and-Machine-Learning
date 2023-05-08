#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self):
        self.input_layer_size = 4
        self.hidden_layer_size = 6
        self.output_layer_size = 3

        self.weights1 = np.ones((self.input_layer_size, self.hidden_layer_size))
        self.weights2 = np.ones((self.hidden_layer_size, self.output_layer_size))

        self.bias1 = np.ones((1, self.hidden_layer_size))
        self.bias2 = np.ones((1, self.output_layer_size))

    def forward(self, X):
        self.z2 = np.dot(X, self.weights1) + self.bias1
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights2) + self.bias2
        y_hat = sigmoid(self.z3)
        return y_hat

    def loss(self, X, y):
        y_hat = self.forward(X)
        loss = 0.5 * np.sum((y - y_hat) ** 2)
        return loss

    def backpropagate(self, X, y):
        y_hat = self.forward(X)

        delta3 = -(y - y_hat) * sigmoid_prime(self.z3)
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.weights2.T) * sigmoid_prime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        dJdb2 = np.sum(delta3, axis=0)
        dJdb1 = np.sum(delta2, axis=0)

        learning_rate = 0.1

        self.weights1 -= learning_rate * dJdW1
        self.weights2 -= learning_rate * dJdW2
        self.bias1 -= learning_rate * dJdb1
        self.bias2 -= learning_rate * dJdb2


nn = NeuralNetwork()


data = [float(input()) for _ in range(7)]


X = np.array([data[:4]])
y = np.array([data[4:]])


y_hat = nn.forward(X)


loss_before_training = nn.loss(X, y)


nn.backpropagate(X, y)


y_hat_updated = nn.forward(X)


loss_after_training = nn.loss(X, y)


print('{:.4f}'.format(loss_before_training))
print('{:.4f}'.format(loss_after_training))

