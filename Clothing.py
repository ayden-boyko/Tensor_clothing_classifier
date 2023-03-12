from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_datasets as tfds
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

import math
import numpy as np
import matplotlib.pyplot as plt

import plot as p
from plot import *

"""loads dataset into 2 sets, training set & testing set"""
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

"""create classes for articles of clothing"""
clothing_names = ['Tshirt/top', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

"""convert pixels from dataset from 0,255 range to 0,1 range"""
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

"""normalizes every dataset so it can be used"""
train_dataset = train_dataset.map(normalize)


#create model with three layers
model = tf.keras.Sequential([
    #input layer flattens 28/28 picture into 1 pixel array of 784 elements
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #128 neurons that take input from first layer and input wieght into hidden params,
    #outputs single value to next layer
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    #10 output neurons, each neuron takes in values from the 128 neurons before it,
    #weighs input according to params and outputs value from [1,0] probability of article beloning to class
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""number of examples"""
BATCH_SIZE = 32
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
"""repeat specifies that dataset will train forever"""
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
"""epoch specifies train 5 times, 60000 training examples means it will train 300000 times"""
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

"""see accuracy on test data"""
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/BATCH_SIZE))
print("accuracy on test dataset:", test_accuracy)

"""model predicts image classification"""
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

"""displays results of particular clothing article, represnted by the num you give function, results are diplayed in probability"""
p.show_results(26, predictions, test_labels, test_images)

