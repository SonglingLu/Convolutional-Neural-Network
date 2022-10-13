import numpy as np
import mnist
from tensorflow import keras

train_images = mnist.train_images()
train_images = (train_images / 255) - 0.5
train_images = train_images.reshape((-1, 28 * 28))
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_images = (test_images / 255) - 0.5
test_images = test_images.reshape((-1, 28 * 28))
test_labels = mnist.test_labels()


layers = [
    keras.layers.Dense(64, activation = 'relu', input_shape = (28 * 28,)),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
]

model = keras.models.Sequential(layers)

model.compile (
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit (
    train_images,
    keras.utils.to_categorical(train_labels),
    epochs=5,
    batch_size=32
)


model.evaluate(
    test_images,
    keras.utils.to_categorical(test_labels)
)

model.save_weights('cnnModel.h5')


predictions = model.predict(test_images[:5])
print(np.argmax(predictions, axis=1))
print(test_labels[:5])