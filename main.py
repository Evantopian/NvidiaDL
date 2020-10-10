import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing import image as image_utils


# returns a "scaled" image given its path.
def load_and_scale_image(image_path):
    image = image_utils.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    return image


# Using matplotlib to display the image.
def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)


# method for making the predictions.
def predict_letter(file_path):
    show_image(file_path)
    image = load_and_scale_image(file_path)
    image = image_utils.img_to_array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255

    prediction = model.predict(image)
    # convert prediction to letter
    predicted_letter = dictionary[np.argmax(prediction)]
    return predicted_letter


# Loading in the model from the previous lab, we can use it to test and predict in this lab.
model = keras.models.load_model('asl_model')

model.summary()
show_image('asl_images/b.png')

image = load_and_scale_image('asl_images/b.png')
plt.imshow(image, cmap='gray')


image = image_utils.img_to_array(image)

# corresponds to 1 image of 28x28 px with 1 color channel.
image = image.reshape(1, 28, 28, 1)

# normalizing data
image = image / 255
prediction = model.predict(image)

# does not contain j or z because in asl they're moving representations of the alphabet.
alphabet = "abcdefghiklmnopqrstuvwxy"
dictionary = {}
for i in range(24):
    dictionary[i] = alphabet[i]

# Testing the model for its predictions
predict_letter("asl_images/a.png")
predict_letter("asl_images/b.png")
