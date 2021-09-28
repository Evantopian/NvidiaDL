# Project idea derived from Nvidia Fundamentals of Deep Learning DLI
# I removed the DataSet from the program because it's too large- you can find it on Kaggle.

import os

import tensorflow.keras as keras
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# removes the annoying "successfully connected with cuda" text.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loading in our data into pandas's dataframe
train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")

# organizing the data - creating new variables called y_train and y_test to match the syntax.
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# Separate our our image vectors
x_train = train_df.values
x_test = test_df.values

# Turn our scalar targets into binary categories
y_train = keras.utils.to_categorical(y_train, 25)
y_test = keras.utils.to_categorical(y_test, 25)

# Normalizing the data
x_train = x_train / 255
x_test = x_test / 255

# Reshape the image data for the convolutional network
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


num_classes = 25

# Creating the convolutional model.
model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation='softmax'))


# Augmenting the data.
datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False)  # Don't randomly flip images vertically


# Fitting the data onto the generator.
datagen.fit(x_train)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])


# Training with Data Augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=32),  # Default batch_size is 32. We set it here for clarity.
          epochs=20,
          steps_per_epoch=len(x_train) / 32,  # Run same number of steps we would if we were not using a generator.
          validation_data=(x_test, y_test))

model.save('asl_model')

'''
# Clears memory
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
'''
