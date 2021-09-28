# Project idea derived from Nvidia Fundamentals of Deep Learning DLI
# I removed the DataSet from the program because it's too large- you can find it on Kaggle.

import os
import tensorflow.keras as keras
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

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

# Separate out our image vectors
x_train = train_df.values
x_test = test_df.values

# Turn our scalar targets into binary categories
y_train = keras.utils.to_categorical(y_train, 25)
y_test = keras.utils.to_categorical(y_test, 25)

# Normalizing our data.
x_train = x_train / 255
x_test = x_test / 255
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

# printable summary of the model
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model.
model.fit(x_train, y_train,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))

# Accuracy come close to 100%, but the validation accuracy seems to be "off"


'''
# Memory Clear

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
'''
