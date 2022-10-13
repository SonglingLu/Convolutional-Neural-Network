from pickle import NEWOBJ_EX
from tkinter.ttk import Sizegrip
import numpy as np

class pool:
    def __init__(self, size):
        self.size = size

    def iterate_regions(self, image):
        # generator for iterating over the conv lyer's output
        y, x, _ = image.shape
        new_y = y // self.size
        new_x = x // self.size

        for i in range(new_y):
            for j in range(new_x):
                patch = image[(i * self.size):(i * self.size + self.size),
                    (j * self.size):(j * self.size + self.size)]
                yield patch, i, j

    def forward(self, input):
        y, x, num_layers = input.shape
        self.cache = input
        self.cache_size = input.shape
        output = np.zeros((y // self.size, x // self.size, num_layers))

        for patch, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(patch, axis=(0, 1))

        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.cache_size)

        for patch, i, j in self.iterate_regions(self.cache):
            y, x, num_layers = patch.shape
            out = np.amax(patch, axis=(0, 1))

            for m in range(y):
                for n in range(x):
                    for l in range(num_layers):
                        if patch[m, n, l] == out[l]:
                            d_L_d_input[i * 2 + m, j * 2 + n, l] = d_L_d_out[i, j, l]
      
        return d_L_d_input
