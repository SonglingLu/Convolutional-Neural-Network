import mnist
import numpy as np
from conv import convolution
from pooling import pool
from softmax import softmax

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

conv3x3 = convolution(8, 3, 3)
pool2 = pool(2)
softmax10 = softmax(13 * 13 * 8, 10)



def forward(image, label):
    output = conv3x3.forward((image / 255) - 0.5)
    output = pool2.forward(output)
    output = softmax10.forward(output)

    # calculate loss and accuracy
    l = - np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0

    return output, l, acc


def train(image, label, lr=.005):
    # pass forward the image
    output, loss, accuracy = forward(image, label)

     # initialize the gradient of loss against output
    grad = np.zeros(10)
    grad[label] = -1 / output[label]

    # backward process for softmax layer
    grad = softmax10.backward(grad, lr)
    # backward process for pooling layer
    grad = pool2.backward(grad)
    # backward process for convolution layer
    conv3x3.backward(grad, lr)

    return loss, accuracy

# training
print('\n--- Training myCNN ---')
for epoch in range(10):
    print('--- Epoch %d ---' % (epoch + 1))

    # shuffle the training data for each training epoch
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    loss = 0
    accuracy = 0
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        l, acc = train(image, label)
        loss += l
        accuracy += acc

        if i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, loss / 100, accuracy)
            )

            loss = 0
            accuracy = 0

# testing
print('\n--- Testing myCNN ---')
loss = 0
accuracy = 0
for i, (image, label) in enumerate(zip(test_images, test_labels)):
    _, l, acc = forward(image, label)
    loss += l
    accuracy += acc


num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', accuracy / num_tests)