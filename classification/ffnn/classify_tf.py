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
import matplotlib.pyplot as plt
"""
LOADING THE DATA
"""
img_width = 32
img_height = 32
train_x_orig, train_y, test_x_orig, test_y, classes = load_random_animals(
    dataset_folder = '/home/palma/opencampus/animals_classification_DL/dataset',
    max_classes=5,
    img_size=(img_width, img_height),
    train_test_split=0.9)

train_y = tf.keras.utils.to_categorical(np.transpose(train_y))
test_y = tf.keras.utils.to_categorical(np.transpose(test_y))
print(f"Data shapes:\nX: {train_x_orig.shape}\nY: {train_y.shape}")

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
image_shape = (img_width, img_height, 3)
model = Sequential([
    # .. add here the layers
    layers.Flatten(input_shape=image_shape),
    layers.Rescaling(1/255),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation ='sigmoid')
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
        learning_rate=0.0001
    ),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = ["accuracy"]
)

epochs = 500


history = model.fit(
    # choose images, labels and epochs
    x = train_x_orig,
    y = train_y,
    validation_data = (test_x_orig, test_y),
    epochs = epochs
)

def plot_training_evolution(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(32, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_training_evolution(history)

"""
EVALUATE

Evaluate on the test images
Make some predictions
"""
model.evaluate(
    # choose test images and labels
    test_x_orig,
    test_y
)

predictions = model.predict(test_x_orig)

def show_errors(test_x_orig, test_y, predictions):
    images = 24
    plt.figure(figsize=(32, 10))
    counter = 1
    for i, (prediction, gt) in enumerate(zip(predictions, test_y)):
        #print(prediction.shape, gt.shape)
        if counter <= images:
            if np.abs(np.argmax(prediction) - np.argmax(gt)) > 0.1:
                plt.title(f'image of {classes[np.argmax(gt)]}')
                plt.subplot(counter // 6 + 1, 6, counter % 6 + 1)
                plt.imshow(test_x_orig[i])
                counter+=1
    plt.show()

show_errors(test_x_orig, test_y, predictions)
