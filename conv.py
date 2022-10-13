import numpy as np

class convolution:
    def __init__(self, num_layers, filter_h, filter_w):
        self.num_layers = num_layers
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.kernel = np.random.randn(num_layers, filter_h, filter_w) / 9

    def iterate_regions(self, image):
        # generator for iterating over the image
        y, x = image.shape

        for i in range(y - self.filter_h + 1):
            for j in range(x - self.filter_w + 1):
                patch = image[i:(i + self.filter_h), j:(j + self.filter_w)]
                yield patch, i, j
    
    def forward(self, input):
        y, x = input.shape
        self.cache = input
        output = np.zeros((y - self.filter_h + 1, x - self.filter_w + 1, self.num_layers))

        for patch, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(patch * self.kernel, axis=(1, 2))

        return output

    def backward(self, d_L_d_out, lr):
        d_L_d_filters = np.zeros((self.num_layers, self.filter_h, self.filter_w))

        for patch, i, j in self.iterate_regions(self.cache):
            for l in range(self.num_layers):
                d_L_d_filters[l] += d_L_d_out[i, j, l] * patch

        self.kernel -= lr * d_L_d_filters
        return None