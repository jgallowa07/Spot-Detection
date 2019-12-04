
"""
File: Helpers.py
Authors Jared Galloway, Nick Wagner, Annie Wang
Date: 11/17/2019

This file is where data is prepared and fed into the respective model
functions.
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

##############################################################################

## Empirical images
empiricalg = "../Data/Empirical/L1-D01-g.bmp"
empiricals = "../Data/Empirical/L1-D01-s.bmp"
empiricalz = "../Data/Empirical/L1-D01-z.bmp"
empirical_output = empirical_prep([empiricalg, empiricals, empiricalz])

## Annotation data
width, height = 1024, 1024
anno_file1 = "../Data/Annotation/annotation_output/L1-D01-g_output.csv"
anno_file2 = "../Data/Annotation/annotation_output/L1-D01-s_output.csv"
anno_file3 = "../Data/Annotation/annotation_output/L1-D01-z_output.csv"
# Create three pixelmaps to use for colocalization       
pixelmap1 = dot_click_annoation_file_to_pixelmap(
    anno_file = anno_file1,
    width = width,
    height = height,
    dot_radius = 2)
pixelmap2 = dot_click_annoation_file_to_pixelmap(
    anno_file = anno_file2,
    width = width,
    height = height,
    dot_radius = 2)
pixelmap3 = dot_click_annoation_file_to_pixelmap(
    anno_file = anno_file3,
    width = width,
    height = height,
    dot_radius = 2)
# colocalization
colocalized_output = colocalization([pixelmap1,pixelmap2,pixelmap3])
# sub_patch annotations
sub_annotations = sub_patch_pixelmap(colocalized_output)
sub_annotations = np.expand_dims(sub_annotations, axis=3)

#for i in range(20):
#    plt.imshow(empirical_output[i,:,:,:])
#    plt.show()
#sys.exit()

x = empirical_output
y = sub_annotations

#print(x.shape)
#print(y.shape)

#sys.exit()

# TODO should probably shuffle before here

num_samples = 2500
width = 32
height = 32
x = np.zeros([num_samples, width, height, 3])
y = np.zeros([num_samples, width, height])
for i in range(num_samples):
    X, Y = generate_simulated_microscopy_sample(
        colocalization = [2] + [0 for _ in range(6)],
        width = width,
        height = height)
    add_normal_noise_to_image(X,0.01)
    x[i] = X
    y[i] = Y
y = np.reshape(y, [num_samples, width, height, 1])

print(x.shape)
print(y.shape)

test_split = int(x.shape[0] * 0.1) 
vali_split = int(x.shape[0] * 0.2)

test_x = x[:test_split,:,:,:]
vali_x = x[test_split:vali_split,:,:,:]
train_x = x[vali_split:,:,:,:]

test_y = y[:test_split,:,:,:]
vali_y = y[test_split:vali_split,:,:,:]
train_y = y[vali_split:,:,:,:]

model = initial_CNN_map(x,y)
print(model.summary())

model.fit(train_x, train_y, 
        validation_data = (vali_x, vali_y),
        epochs = 50)

# TODO: Try and figure out wtf is going on :)
