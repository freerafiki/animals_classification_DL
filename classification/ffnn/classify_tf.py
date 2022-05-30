"""
Animals classification using Tensorflow and Keras.
"""

from load_animals.load_random_animals import load_random_animals # loading the animals
import pdb # debugging
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np


"""
LOADING THE DATA
"""
train_x_orig, train_y, test_x_orig, test_y, classes = load_random_animals(
    dataset_folder = '/home/palma/opencampus/animals_classification_DL/dataset',
    max_classes=3,
    img_size=(64, 64),
    train_test_split=0.9)

"""
CREATE THE MODEL
there are several layers, that can be seen here: https://keras.io/api/layers/
We will use the Sequential API to create a simple model.
This allows us to declare layers one after the others.

For example
----
To convert images to array (from 2D to 1D), there is a
layers.Flatten(input_shape=(img_height, img_width, 3))
----
The fully connected layer is
layers.Dense(num_of_neurans, activation='..') # the activations are the usuals
----
Dropout can be added with
layers.Dropout(rate)
----
Rescaling the images could be done with
layers.Rescaling(scaling_factor, input_shape=(img_height, img_width, 3))
"""
model = Sequential([
    # .. add here the layers
])

# print out the model
model.summary()

"""
TRAINING

First, choose an optimizer, a loss and some metrics.
Then feed the model with the data
"""
model.compile(
    # choose optimizer, loss and metrics
)

epochs =

model.fit(
    # choose images, labels and epochs
)

"""
EVALUATE

Evaluate on the test images
Make some predictions
"""
model.evaluate(
    # choose test images and labels
)

predictions = probability_model.predict(test_images)
