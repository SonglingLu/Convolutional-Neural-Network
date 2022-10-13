import numpy as np
from neuron import *
from matplotlib import pyplot as plt

# mse lose function
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class myNN:

    def __init__(self):
        w1 = np.random.normal()
        w2 = np.random.normal()
        w3 = np.random.normal()
        w4 = np.random.normal()
        w5 = np.random.normal()
        w6 = np.random.normal()

        b1 = np.random.normal()
        b2 = np.random.normal()
        b3 = np.random.normal()

        # neurons for hiddden layer
        self.n1 = Neuron(np.array([w1, w2]), b1)
        self.n2 = Neuron(np.array([w3, w4]), b2)
        # final output neuron
        self.o1 = Neuron(np.array([w5, w6]), b3)


    def forward(self, input):
        out_n1 = self.n1.forward(input)
        out_n2 = self.n2.forward(input)

        # Take outputs from n1 and n2 and pass them to o1
        out_o1 = self.o1.forward(np.array([out_n1, out_n2]))

        return out_o1
    

    def update(self, x, y_true, lr):
        total_n1 = self.n1.total(x)
        sig_n1 = sigmoid(total_n1)

        total_n2 = self.n2.total(x)
        sig_n2 = sigmoid(total_n2)

        total_o1 = self.o1.total(np.array([sig_n1, sig_n2]))
        sig_o1 = sigmoid(total_o1)
        y_pred = sig_o1

        # Calculate partial derivatives     
        # Neuron o1
        d_o1 = -2 * (y_true - y_pred)
        d_w5 = sig_n1 * deriv_sigmoid(total_o1)
        d_w6 = sig_n2 * deriv_sigmoid(total_o1)
        d_b3 = deriv_sigmoid(total_o1)

        # Neuron h1
        d_n1 = self.o1.weights[0] * deriv_sigmoid(total_o1) * d_o1
        d_w1 = x[0] * deriv_sigmoid(total_n1)
        d_w2 = x[1] * deriv_sigmoid(total_n1)
        d_b1 = deriv_sigmoid(total_n1)

        # Neuron h2
        d_n2 = self.o1.weights[1] * deriv_sigmoid(total_o1) * d_o1
        d_w3 = x[0] * deriv_sigmoid(total_n2)
        d_w4 = x[1] * deriv_sigmoid(total_n2)
        d_b2 = deriv_sigmoid(total_n2)

        # Update weights and biases
        self.n1.update(lr, d_n1, np.array([d_w1, d_w2]), d_b1)
        self.n2.update(lr, d_n2, np.array([d_w3, d_w4]), d_b2)
        self.o1.update(lr, d_o1, np.array([d_w5, d_w6]), d_b3)

        return


    def train(self, data, y):
        lr = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, y):
                self.update(x, y_true, lr)

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.forward, 1, data)
                loss = mse_loss(y, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
                loss_epoch[0].append(loss)
                loss_epoch[1].append(epoch)


# Define dataset
data = np.array([
    [-2, -1],    # Alice
    [25, 6],     # Bob
    [17, 4],     # Charlie
    [-15, -6], # Diana
])
all_y_trues = np.array([
    1, # Alice
    0, # Bob
    0, # Charlie
    1, # Diana
])

# Train the network
loss_epoch = [[], []]
network = myNN()
network.train(data, all_y_trues)

# plot the result
plt.plot(loss_epoch[1], loss_epoch[0])
plt.show()