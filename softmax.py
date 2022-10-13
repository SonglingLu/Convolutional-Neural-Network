import numpy as np

class softmax:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)

    def forward(self, input):
        # cache the input
        self.cache_size = input.shape
        input = input.flatten()
        self.cache = input

        input_size, output_size = self.weights.shape

        # cache the result of exp(w*x + b)
        result = np.exp(np.dot(input, self.weights) + self.biases)
        self.cache_result = result
        return result / np.sum(result, axis=0)

    def backward(self, d_L_d_out, lr):
        # the correct label is c and others are k
        for i, grad in enumerate(d_L_d_out):
            # only output(c) will effect the loss, meaning only its gradient is nonzero
            if grad == 0:
                continue

            # sum of exp(w*x + b)
            exp_sum = np.sum(self.cache_result)

            # gradients of output againts f(x) for labels k
            d_out_d_f = - self.cache_result[i] * self.cache_result / (exp_sum ** 2)
            # gradients of output againts f(x) for the correct label c
            d_out_d_f[i] = self.cache_result[i] * (exp_sum - self.cache_result[i]) / (exp_sum ** 2)

            # gradients of f(x) against w, b, input
            d_f_d_w = self.cache
            d_f_d_b = 1
            d_f_d_input = self.weights

            # gradient of loss against f(x)
            d_L_d_f = grad * d_out_d_f

            # gradients of loss against w, b, input
            d_L_d_w = d_f_d_w[:, np.newaxis] @ d_L_d_f[np.newaxis]
            d_L_d_b = d_f_d_b * d_L_d_f
            d_L_d_input = d_f_d_input @ d_L_d_f

            # update w, b
            self.weights -= lr * d_L_d_w
            self.biases -= lr * d_L_d_b

            return d_L_d_input.reshape(self.cache_size)