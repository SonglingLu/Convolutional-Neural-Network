import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        total = self.total(inputs)
        return sigmoid(total)

    def total(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def update(self, lr, d_l, d_w, d_b):
        self.weights -= lr * d_l * d_w
        self.bias -= lr * d_l * d_b
        return