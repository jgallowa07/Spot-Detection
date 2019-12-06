
"""
File: Helpers.py
Authors Jared Galloway, Nick Wagner, Annie Wang
Date: 11/17/2019

This file is for early experimentation of neural net work on various functions.
"""

##############################################################################

import os
import sys
sys.path.insert(0,"../")

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
from networks import *
import time
from DataPrep.helpers  import *


# Hard Dataset

params = {"num_samples":3500,
            "width":64,
            "height":64,
            "coloc_thresh":3,
            "colocalization":[1 for _ in range(7)],
            "noise":0.2}

x, y = simple_simulator(**params)

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

# fit a model!

model = deeper_direct_CNN(x,y)
print(model.summary())

model.fit(train_x, train_y, 
        validation_data = (vali_x, vali_y),
        epochs = 50)

pred = model.predict(test_x)
pred.dump("hard_y.out")
test_x.dump("hard_x.out")
