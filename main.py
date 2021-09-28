# Project idea derived from Nvidia Fundamentals of Deep Learning DLI
# removes the annoying "successfully connected with cuda" text.

# I removed the DataSet from the program because it's too large- you can find it on Kaggle.

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
By importing Pandas, you are now able to load and work with the data. Pandas is a tool
used for manipulating data - reads the csv files into a format called Pandas Dataframe.
'''
import pandas as pd

train_df = pd.read_csv("asl_data/sign_mnist_train.csv")
test_df = pd.read_csv("asl_data/sign_mnist_test.csv")

# prints the first few rows of the dataframe
train_df.head()

# organizing the data - creating new variables called y_train and y_test to match the syntax.
y_train = train_df['label']
y_test = test_df['label']
# delete previous labels from the dataframe - it is redundant to keep.
del train_df['label']
del test_df['label']

# storing the data
x_train = train_df.values
x_test = test_df.values

# 27,455 images for training (784px) & their corresponding  labels
x_train.shape
y_train.shape

# 7,172 images for testing (784px) & their corresponding  labels
x_test.shape
y_test.shape


# importing matplotlib to render an image to see the value of it.
import matplotlib.pyplot as plt

plt.figure(figsize=(40, 40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]

    image = row.reshape(28, 28)
    plt.subplot(1, num_images, i + 1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')

# normalizing the data
x_train = x_train / 255
x_test = x_test / 255

# categorically encoding the labels.
import tensorflow.keras as keras

num_classes = 25

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# create the model.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# print out summary of the model
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# begin training
model.fit(x_train, y_train,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))

# both the validation accuracy and accuracy reached close to 100%. (~99%)
