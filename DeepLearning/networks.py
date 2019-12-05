import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def initial_CNN_map(x,y):
    """
    This function is in charge of taking in the testing and training data and
    returning a model object that we can use within train.py. This model will
    be creating a final map output where 1s are synapses and 0s are not.

    params:
    
    x - This should be the input images in the shape (batch, width, height, channels)
    y - This should be the output targets

    This function returns a TensorFlow model object
    """
    
    
    input_layer = tf.keras.layers.Input(shape = x.shape[1:]) # Shape here does not include the batch size

    # cnn layer
    cnn_layer1 = tf.keras.layers.Convolution2D(64, (3,3), padding='same')(input_layer)
    layer1_activation = tf.keras.layers.LeakyReLU()(cnn_layer1)
    output_layer = tf.keras.layers.Convolution2D(1, (3,3), padding='same', activation="sigmoid")(layer1_activation)
    cpd_model=tf.keras.models.Model(input_layer,output_layer)

    cpd_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
        
    return cpd_model

def cnn_to_probability_map_sequential(x,y):
    pass
    


def initial_CNN_count(data):
    """
    This function is in charge of taking in the testing and training data and
    returning a model object that we can use within train.py. This model will
    be creating a final output that is just a count of synapses detected.

    params:

    This function returns a TensorFlow model object
    """
    pass
