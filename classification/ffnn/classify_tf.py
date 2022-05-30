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
    max_classes=6,
    img_size=(256, 256),
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
layers.Dense(num_of_neurons, activation='..') # the activations are the usuals
----
Dropout can be added with
layers.Dropout(rate)
----
Rescaling the images could be done with
layers.Rescaling(scaling_factor, input_shape=(img_height, img_width, 3))
"""
random_image = train_x_orig[100]
np.max(random_image)
image_shape = (256, 256, 3)
model = Sequential([
    # .. add here the layers
    layers.Flatten(input_shape=image_shape),
    layers.Rescaling(1/256 ),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),

    layers.Dense(3, activation ='sigmoid')
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
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.005,
        # beta_1=0.9,
        # beta_2=0.999,
        # epsilon=1e-07,
        # amsgrad=False,
        name='Adam'
    ),
    loss = tf.keras.losses.CategoricalCrossentropy(
        # from_logits=False,
        # label_smoothing=0.0,
        # axis=-1,
        # name='categorical_crossentropy'
    ),
    metrics = ["acc"]
)

epochs = 100

train_x_orig.shape
train_y.shape
train_y_t = np.transpose(train_y)
train_y_t.shape
categorical_y = tf.keras.utils.to_categorical(train_y_t)
categorical_y
model.fit(
    # choose images, labels and epochs
    train_x_orig,
    categorical_y,
    epochs = epochs
)

"""
EVALUATE

Evaluate on the test images
Make some predictions
"""
model.evaluate(
    # choose test images and labels
    test_x_orig,
    tf.keras.utils.to_categorical(np.transpose(test_y))
)

predictions = model.predict(test_x_orig)
predictions
