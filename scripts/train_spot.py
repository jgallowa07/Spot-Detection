
"""
File: train_spot.py
Authors Jared Galloway, Nick Wagner, Annie Wang
Date: 11/17/2019

This file is for early experimentation of neural net work on simulated
and empirical images.
"""

##############################################################################

import os
import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
from networks import *
import time
from helpers import *
from simulator import *


output = "l2_s05_p45_b20"
epochs = 50

# Hard Dataset

params = {"num_samples":10,
            "width":64,
            "height":64,
            "coloc_thresh":2,
            "num_dots_n": 5,
            "num_dots_p": 0.85,
            "radii_n":4,
            "radii_p":0.65,
            "spot_noise":0.2,
            "point_noise":0.2,
            "background_noise":0.15}

x, y = simulate_single_layers(**params)

print(f"simulated data set x has shape: {x.shape}")
print(f"simulated data set y has shape: {y.shape}")

# cut data set up into train, validation, and testing.

test_split = int(x.shape[0] * 0.1) 
vali_split = int(x.shape[0] * 0.2)

test_x = x[:test_split,:,:,:]
vali_x = x[test_split:vali_split,:,:,:]
train_x = x[vali_split:,:,:,:]

test_y = y[:test_split,:,:,:]
vali_y = y[test_split:vali_split,:,:,:]
train_y = y[vali_split:,:,:,:]

# choose a model!
model = deeper_direct_CNN(x,y)
print(model.summary())

# fit the model
model.fit(train_x, train_y, 
        validation_data = (vali_x, vali_y),
        epochs = epochs)

pred = model.predict(test_x)
pred.dump(f"./output/pred_{output}.out")
test_x.dump(f"./output/x_{output}.out")
test_y.dump(f"./output/y_{output}.out")
