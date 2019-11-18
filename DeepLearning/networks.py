import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def initial_CNN_map(data):
    """
    This function is in charge of taking in the testing and training data and
    returning a model object that we can use within train.py. This model will
    be creating a final map output where 1s are synapses and 0s are not.

    params:

    This function returns a TensorFlow model object
    """
    
    

    input_layer=tf.keras.layers.Input( shape=cpd.X_train.shape[1:] ) # Shape here does not include the batch size 

    # cnn layer
    cnn_layer1=tf.keras.layers.Convolution2D(64, (24,24),strides=6,padding='same')(input_layer)
    cnn_activation=tf.keras.layers.LeakyReLU()(cnn_layer1)

    # maxpool
    max_pool=tf.keras.layers.MaxPooling2D(pool_size=(8, 8))(cnn_activation)

    # cnn layer
    cnn_layer1=tf.keras.layers.Convolution2D(32, (4,4),strides=2,padding='same')(cnn_activation)
    cnn_activation=tf.keras.layers.LeakyReLU()(cnn_layer1)

    # maxpool
    max_pool=tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(cnn_activation)

    # cnn layer
    cnn_layer1=tf.keras.layers.Convolution2D(16, (4,4),strides=2,padding='same')(cnn_activation)
    cnn_activation=tf.keras.layers.LeakyReLU()(cnn_layer1)

    max_pool=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cnn_activation)

    # dropout layer
    dropout=tf.keras.layers.Dropout(0.15)(max_pool)

    ## Here is our magic layer to turn image data into something a dense layer can use
    flat_input=tf.keras.layers.Flatten()(dropout)#Dense layers take a shape of ( batch x features)


    ##
    hidden_layer1=tf.keras.layers.Dense(10)(flat_input)    
    hidden_layer_activation=tf.keras.layers.LeakyReLU()(hidden_layer1)

    dropout_layer=tf.keras.layers.Dropout(0.15)(hidden_layer_activation)

    dropout_output=tf.keras.layers.Dense(1)(dropout_layer)

    output_layer=tf.keras.layers.Dense(1,activation='sigmoid')(dropout_output)
    cpd_model=tf.keras.models.Model(input_layer,output_layer)



    # optimizer
    optimizer=tf.keras.optimizers.Adam(lr=1e-3)

    cpd_model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    def initial_CNN_count(data):
        """
        This function is in charge of taking in the testing and training data and
        returning a model object that we can use within train.py. This model will
        be creating a final output that is just a count of synapses detected.

        params:

        This function returns a TensorFlow model object
        """
