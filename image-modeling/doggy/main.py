# Project idea derived from Nvidia Fundamentals of Deep Learning DLI
# I removed the DataSet from the program because it's too large- you can find it on Kaggle.
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions

# removes the annoying "successfully connected with cuda" text.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)


def load_and_process_image(image_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)

    # Load in the image with a target size of 224,224
    image = image_utils.load_img(image_path, target_size=(224, 224))
    # Convert the image from a PIL format to a numpy array
    image = image_utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1, 224, 224, 3)
    # Preprocess image to align with original ImageNet dataset
    image = preprocess_input(image)
    # Print image's shape after processing
    print('Processed image shape: ', image.shape)
    return image


def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:', decode_predictions(predictions, top=3))


def doggy_door(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    preds = model.predict(image)
    if 151 <= np.argmax(preds) <= 268:
        print("Doggy come on in!")
    elif 281 <= np.argmax(preds) <= 285:
        print("Kitty stay inside!")
    else:
        print("You're not a dog! Stay outside!")


# Loads in a pre - trained model from the VGG16 Network which is commonly used.
model = VGG16(weights="imagenet")

# displays an img.
show_image("doggy_door_images/happy_dog.jpg")

# processes an image to display
processed_image = load_and_process_image("doggy_door_images/brown_bear.jpg")

# play around with the images to check your pre-trained model's prediction capabilities
readable_prediction("doggy_door_images/sleepy_cat.jpg")
readable_prediction("doggy_door_images/brown_bear.jpg")
readable_prediction("doggy_door_images/sleepy_cat.jpg")
