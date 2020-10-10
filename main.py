'''
Assessment:
Congratulations on going through today's course! Hopefully you've learned some valuable skills along the way.
Now it's time to put those skills to the test. In this assessment you will train a new model that is able to
recognize fresh and rotten fruit. You will need to get the model to a validation accuracy of 92% in order to
pass the assessment, though we challenge you to do even better if you can. You will have the use the skills
that you learned in the previous exercises. Specifically we suggest you use some combination of transfer learning,
data augmentation, and fine tuning. Once you have trained the model to be at least 92% accurate on the test dataset,
you will save your model, and then assess its accuracy. Let's get started!
'''

# GOAL: Get a validation accuracy above 92% for certification on the fundamentals of DL.

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# downloading the VGG16 Model
base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(224, 224, 3),
    include_top=False)

# Freeze base model
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)  # flattening the layer

# Add six dense layers because there are 2 variations of 3 types of fruits.
outputs = keras.layers.Dense(6)(x)

# Combine input & output to make the model.
model = keras.Model(inputs, outputs)

# Checking the model info
model.summary()

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

# augment the data.
datagen = ImageDataGenerator(
    samplewise_center=True,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

# load and iterate training dataset
train_it = datagen.flow_from_directory('fruits/train/',
                                       target_size=(224, 224),
                                       color_mode='rgb',
                                       class_mode='binary',
                                       batch_size=8)
# load and iterate test dataset
test_it = datagen.flow_from_directory('fruits/test/',
                                      target_size=(224, 224),
                                      color_mode='rgb',
                                      class_mode='binary',
                                      batch_size=8)

# unfreeze the model
base_model.trainable = True

# training the model
model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=20)

# evaluate the model's accuracy and validation accuracy.
model.evaluate(test_it, steps=test_it.samples / test_it.batch_size)
# My validation accuracy in my Jupyter notebook was a 98.87%. Percent to pass: 92%


'''
# run to see whether 
from run_assessment import run_assessment
run_assessment(model, test_it)


# Memory Clear:
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
'''
