# Project idea derived from Nvidia Fundamentals of Deep Learning DLI
# removes the annoying "successfully connected with cuda" text.
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# loads keras dataset module from MNIST
from tensorflow.keras.datasets import mnist

'''
x_train = the images used for training the neural network.
y_train = the correct labels for the x_train images. (Used for evaluation)

x_test = The images set aside for testing the performance of the model after it has been trained.
y_test = The correct labels for the x_test images. (Used for evaluation)
'''

# The data, split between train and test sets
# Partitions 60,000 photos to train, and 10,000 to test.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Checking the shape in the beginning to have a better understanding.
x_train.shape
x_test.shape

# images are 28x28 in gray scale. Values range from 0-255, other values are represented by:
x_train.dtype
x_train.min()
x_train.max()
x_train[0]

# importing matplotlib to render an image to see the value of it.
import matplotlib.pyplot as plt

image = x_train[0]
plt.imshow(image, cmap='gray')

# the answer is the correct labels found in y_train
y_train[0]

# Flattening the Image data -> from a 2D array, ex[28][28], to ex[784]. This simplifies things.
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train.shape

# DL models are better at dealing with floating values between 0.0-1.0, so we normalize them.
# Pixel color range between 0-255, so you divide by 255.
x_train = x_train / 255
x_test = x_test / 255

# Checking to see if normalized correctly.
x_train.dtype
x_train.min()
x_train.max()

# Categorically encoding the values so that the neural network understands that guessing 4 is just as bad as 9, if 5.
import tensorflow.keras as keras

num_categories = 10

y_train = keras.utils.to_categorical(y_train, num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)
# Shows the first 10 values of the training labels that have now been categorically encoded.

y_train[0:9]

# Creating the model.
from tensorflow.keras.models import Sequential

# Instantiating an instance of a model with a series of layers with data passing in sequences.
model = Sequential()

# The neurons will be densely connected, so Dense will represent how weights and the previous neurons effect the next.
from tensorflow.keras.layers import Dense

# units = number of neurons in a layer, 512 is a good beginning number.
# relu = rectifier in a neural network
# softmax = normalized exponential function
model.add(Dense(units=512, activation='relu', input_shape=(784,))) # input shape = type of incoming data.
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# readable summary of the model.
model.summary()

# compiles the model.
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# begins the training.
history = model.fit(x_train, y_train,
                    epochs=5,
                    verbose=1,
                    validation_data=(x_test, y_test))

# epochs = a full run through in the training set.

# both the validation accuracy and accuracy reached close to 100%. (~99%)
'''
# clears gpu memory
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
'''