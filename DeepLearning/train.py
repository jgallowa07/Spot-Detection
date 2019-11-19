
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
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt
from networks import *

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

x = empirical_output
y = sub_annotations

# should probably shuffle before here

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


#print(test_split)
#print(vali_split)










#print(len(empirical_output))
#fig, ax = plt.subplots(3)
#for i in range(3):
#    ax[i].imshow(np.array(empirical_output[0][0])[:,:,i])
#print(np.all(np.equal(np.array(empirical_output[0][0])[:,:,1],np.array(empirical_output[0][0])[:,:,2])))
#print(len(empirical_output[0]))
#print("")
#print(sub_annotations.shape)
### This is where I would break up the information into training/testing
