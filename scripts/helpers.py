
"""
File: Helpers.py
Authors Jared Galloway, Nick Wagner, Annie Wang
Date: 11/20/2019

This file contains all helpful python
functions for scripts included in synapse detection.
"""

###

import os
import sys

import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import circle
from mpl_toolkits.mplot3d import Axes3D

###

# UNDER CONSTRUCTION
def dot_click_annoation_file_to_pixelmap(anno_file,
                                                width,
                                                height,
                                                dot_radius):
    """
    This function takes in an csv image anno which has the
    row format: filepath, top x, top y, bottom x, bottom y. no header.
    
    
    :param: anno_file <string>  - path to annotation file

    :param: width <int> - width of the sample 
    
    :param: height <int> - height of the sample
    
    :param: dot_radius <int> - radius of bumps
 
    EX Data in file:
    L1-D01-g.bmp,583,247,591,255,synapse
    L1-D01-g.bmp,589,256,597,264,synapse
    L1-D01-g.bmp,559,269,567,277,synapse
    L1-D01-g.bmp,592,267,600,275,synapse
    L1-D01-g.bmp,635,264,643,272,synapse
    L1-D01-g.bmp,607,281,615,289,synapse
    L1-D01-g.bmp,595,284,603,292,synapse

    :return: pixelmap <ndarray> - the function will return a 2-dimentional binary 
        numpy ndarray object with the shape (width,height). 1's represent annotated 
        images
    """

    # initialize a pixelmap
    pixelmap = np.zeros([width,height], dtype=np.uint8)

    # draw ones using skimage circle()
    # https://scikit-image.org/docs/dev/api/skimage.draw.html
    # skimage.draw.circle
    for line in open(anno_file,"r"):

        # [filepath, top x, top y, bottom x, bottom y]
        anno = line.strip().split(',')
       
        # these should be the original x, and y's when given a bbox. 
        mid_x = (int(anno[1]) + int(anno[3])) // 2
        mid_y = (int(anno[2]) + int(anno[4])) // 2

        assert(0 <= mid_x <= width)  # annotation out of range
        assert(0 <= mid_y <= height) # annotation out of range

        # yay skimage!
        rr,cc = circle(mid_x, mid_y, dot_radius)

        # cut out extreneous indices.
        count = 0
        for i in range(len(rr)):
            if (rr[i-count] >= 1024 or rr[i-count] < 0):
                rr = np.delete(rr, i-count)
                cc = np.delete(cc, i-count)
                count += 1
            if (cc[i-count] >= 1024 or cc[i-count] < 0):
                rr = np.delete(rr, i-count)
                cc = np.delete(cc, i-count)
                count += 1
         
        # set pixel values.
        pixelmap[rr, cc] = 1

    return pixelmap

###

def synquant_to_pixelmap(filename, size = 1024):
    from read_roi import read_roi_zip
    """
    This function should take in the output from SynQuant
    https://www.biorxiv.org/content/10.1101/538769v1
    and convert it to a pixelmap 

    Utilizes a package called read_roi to load in the JSON file 
    that is output from SynQuant

    :param: filename <string> - path to the desired SynQuant output
        file to be read in and converted to a pixelmap

    :param: size <int> - value to define the SIZExSIZE area that was
        fed into the synquant program

    :return: map <numpy array> - returns a SIZExSIZE numpy array that 
        has 1s in all of the pixel (x,y) locations that came from the
        output of SynQuant. 0s everywhere else.
    """

    # read in the JSON style SynQuant output file into roi variable
    roi = read_roi_zip(filename)

    # initialize blank lists for the x and y coordinates that come from
    # the SynQuant output
    xcoord=[]
    ycoord=[]

    # loop that goes through all of the synquant output and pulls out the 
    # x and y coordinates and stores them in the respective lists
    for i in roi.keys():
        xcoord=np.append(xcoord,(roi[i]['x']))
        ycoord=np.append(ycoord,roi[i]['y'])
    # convert lists to integer values
    xcoord=xcoord.astype(int)
    ycoord=ycoord.astype(int)

    # initialize SIZExSIZE numpy array of all zeros 
    map = np.zeros((size,size),dtype=int)

    # loop through the length of the coordinate lists and add each combination
    # of x and y values to map numpy array
    for i in range(len(xcoord)):
        map[xcoord[i]-1,ycoord[i]-1]+=1

    return map

###

def colocalization(pixelmap_list):
    """
    This function takes in a list of pixelmaps and performs bitwise operations 
    to find the spots that have pixels in common (colocalization)

    This function is dependent on the length of the pixelmap_list. If the length is
    just two then it will compute the bitwise-AND, and find the spots of colocalization
    for those two images. If the length is three then it will find all three sets of 
    image colocalization and bitwise-OR those together.

    :param: pixelmap_list <list> - list of paths to pixelmaps in which colocalization
        is desired to be computed between

    :return: COLOCALIZED <ndarray> - This function will return a 2-dimentional binary 
    numpy ndarray object with the shape (1024,1024). 1's represent any point of 
    colocalization between the images.
    """
    pixelmap_list = np.array(pixelmap_list)

    SHAPE = pixelmap_list[0].shape
    COLOCALIZED = np.zeros((SHAPE[0],SHAPE[1]), dtype=np.uint8)

    # if(len(pixelmap_list) == 1):   # The user did not provide enough information to calculate the colocalization
    #     return "Please provide a list of at least two pixelmaps."
    assert(len(pixelmap_list) >= 2)
    
    
    # Case where two pixelmaps are provided
    if(len(pixelmap_list) == 2):
        if(pixelmap_list[0].shape != pixelmap_list[1].shape):
            return "Please provide pixelmaps with the same dimensions"
        
        # performs a bitwise-AND to keep only the pixels that share a spot of colocalizaiton
        COLOCALIZED = np.bitwise_and(pixelmap_list[0], pixelmap_list[1])
    

    # Case where three pixelmaps are provided
    if(len(pixelmap_list) == 3):
        if(pixelmap_list[0].shape != pixelmap_list[1].shape or pixelmap_list[0].shape != pixelmap_list[2].shape):
            return "Please provide pixelmaps with the same dimensions"

        # performs three bitwise-ANDs to keep only the pixels that share a spot of colocalizaiton
        coloalized1 = np.bitwise_and(pixelmap_list[0], pixelmap_list[1])
        coloalized2 = np.bitwise_and(pixelmap_list[0], pixelmap_list[2])
        coloalized3 = np.bitwise_and(pixelmap_list[1], pixelmap_list[2])

        # performs two bitwise-ORs to combine all three sets of colocalization
        COLOCALIZED = np.bitwise_or(coloalized1, coloalized2)
        COLOCALIZED = np.bitwise_or(COLOCALIZED, coloalized3)

    
    return COLOCALIZED

###

def sub_patch_pixelmap(image_pixelmap, size=32, height=(256,1024), width=(256,768)):
    """
    This function allows the user to break up the given pixelmap into sub-patches. The 
    user can specify the area they would like to sub-patch as well as the size of
    the patch they would like to grab.

    :param: image_pixelmap <ndarray> - numpy 2d array corresponding to the pixelmap 
        to be sub-patched

    :param: size <int> - the SIZExSIZE chunk to be grabbed

    :param: height <tuple> - tuple specifying the y start and stop positions (start, stop)  
            DEFAULT: (256,1024)

    :param: width <tuple> - tuple specifying the x start and stop positions (start, stop)  
            DEFAULT: (256,768)
    
    :return: SUB_IMAGES <ndarray> - This function will return a numpy ndarray of 
        SIZExSIZE 2-dimentional binary numpy ndarrays
    """

    SUB_IMAGES = []    # initialize an array for holding the sub images

    for i in range(height[0],height[1],size):  # this for loop isolates only the region of the image specified in the parameters
        for j in range(width[0],width[1],size):
            temp_array = image_pixelmap[i:i+size,j:j+size] # grabbing SIZExSIZE chunks and storing them in an array
            SUB_IMAGES.append(temp_array)
    
    SUB_IMAGES = np.array(SUB_IMAGES)

    return SUB_IMAGES

###

def empirical_prep(list_of_paths, size=32, height=(256,1024), width=(256,768)):
    """
    This function allows the user to break up the empirical images into sub images for 
    training or testing. The user can give as many paths to images as they would like in
    the list_of_paths parameter. The user can specify the area they would like to sub-image 
    as well as the size of the chunk they would like to grab.

    :param: list_of_paths <list> - list of strings corresponding to the paths of 
        the images wanting to be sub-imaged

    :param: size <int> - the SIZExSIZE chunk to be grabbed

    :param: height <tuple> - tuple specifying the y start and stop positions (start, stop)  
            DEFAULT: (256,1024)

    :param: width <tuple> - tuple specifying the x start and stop positions (start, stop)  
            DEFAULT: (256,768)

    :return: sub_empirical <ndarray> -This function will return a 4D numpy ndarray, of 
        which contain all of the sub-images for that given empirical image. There will be 
        one item in the list for each empirical image given
    """

    sub_empirical = []

    # loop through length of list_of_paths
    for num in range(len(list_of_paths)):

        # using pillow .crop function to sub-patch within the image
        pillow_opened_image = Image.open(list_of_paths[num])
        temp_sub_images = []

        # crops SIZExSIZE chunks from the image and appends them as numpy arrays
        # to sub_empirical list
        # this for loop isolates only the region of the image specified in the 
        # parameters
        for i in range(height[0],height[1],size):  
            for j in range(width[0],width[1],size):
                # grabbing SIZExSIZE chunks and storing them in an array
                temp_pic = pillow_opened_image.crop((i,j,i+size,j+size)) 
                temp_pic = np.array(temp_pic)[:,:,0]
                temp_sub_images.append(temp_pic)
        
        # for each item in list_of_paths save the temp images as one of the
        # dimensions of the sub_empirical list
        temp_sub_images = np.array(temp_sub_images)
        sub_empirical.append(temp_sub_images)
    
    # turn list into numpy ndarray
    sub_empirical = np.array(sub_empirical)
    
    # here, we are re-arranging the axes so 
    # we have (batch, height, width, channels)
    sub_empirical = np.swapaxes(sub_empirical,0,1)
    sub_empirical = np.swapaxes(sub_empirical,1,2)
    sub_empirical = np.swapaxes(sub_empirical,2,3)
    

    return sub_empirical

###

def f1_score(pixelmap1, pixelmap2):
    """
    this function will take two pixelmaps (2d-ndarray)
    and return the f1 score defined as:
    
    f1 = 2 / ((1 / precision) + (1/ recall)).
    precision = true positive / (true positive + false positiv)
    recall = true positive / (true positive + false negative)

    :param: pixelmap1, pixelmap2 <ndarray> - two pixelmaps of the same 
        size which we would like to evaluate and get a metric of how they compare

    :return: 2/((1/precision) + (1/recall)) <float> - This function will return 
        a float value between 0 and 1. 0 being the worst the model could be doing 
        and 1 being the best.
    """
    assert(pixelmap1.shape == pixelmap2.shape)
    assert(pixelmap1.dtype == np.int and pixelmap2.dtype == np.int)

    # sum the number of 1s in each of the three combinations 
    # to be able to calculate the two metrics
    true_positive = np.sum(np.bitwise_and(pixelmap1, pixelmap2))
    false_positive = np.sum(np.bitwise_and(pixelmap1, ~pixelmap2))
    false_negative = np.sum(np.bitwise_and(~pixelmap1, pixelmap2))

    # calculate the two metrics described above
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return 2/((1/precision) + (1/recall))

###

def tensor_to_3dmap(tensor, out = None, cmap = "bone"):
    """
    A function which takes in a 2D numpy array and produces 
    a heatmap.

    if a filename is given to out then it will save the fig,
    otherwise it will attempt to open the png with matplotlib.
    """

    X = np.arange(0, tensor.shape[0])
    Y = np.arange(0, tensor.shape[1])
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, tensor, rstride=1, 
        cstride=1, cmap=cmap, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if out == None:
        plt.show()
    else:
        plt.savefig(out)

    return None

###

def f1_score_pixel_v_prob(prediction, target, threshold = 0.7):
    """
    Take in a pixelmap target, and a probability map 
    prediction from our network and return the f1 score. 

    Pixels with probablility > threshold will be considered synapses

    :param: prediction <ndarray> - probability map that is the output
        from our model

    :param: target <ndarray> - pixelmap target that shows the exact
        locations of where the synapses should have been predicted

    :param: threshold <float> - value that represents the prediction
        probability that has to be given in order for it to
        actually be considered a correct guess

    :return: np.mean(agg_fscore) <float> - the mean value of all of the
        f1_scores when comparing the prediction to the target
    """
    assert(prediction.shape == target.shape)    

    # initialize numpy array with zeros to the size of th
    agg_fscore = np.zeros(prediction.shape[0])

    # loops the amount of times as the length of the first dimension
    # calculates an f1_score for each loop by setting the prediction
    # values to 1 if they are above the threshold and 0 otherwise
    # then utilizing the f1_score method
    for i in range(len(prediction)):
        pred_pm = np.squeeze(prediction[:])
        targ = np.squeeze(target[:])
        pred_pm[pred_pm > threshold] = 1
        pred_pm[pred_pm != 1] = 0
        agg_fscore[i] = f1_score(pred_pm.astype(np.int), targ.astype(np.int))

    # return the mean of all of these f1_score calculations
    return np.mean(agg_fscore)




def crop_empirical(path, x, y, size):
    """
    The purpose of this funciton is to be able to crop a single layer of an 
    empirical image to use either as background for simulation or for predicitons
    within the model. 

    :param: path <string> - path to the desired image to crop

    :param: x <float> - starting x position to crop from

    :param: y <float> - starting y position to crop from

    :param: size <int> - the SIZExSIZE chunk to crop from the given image

    :return: cropped_image <ndarray> - numpy ndarray containing the
        cropped section of the image
    """

    # using pillow to load in the image given by the path
    pillow_opened_image = Image.open(path)
    opened_image = np.array(pillow_opened_image)

    # asserts to make sure the coordinates of the full crop are within
    # the valid range
    assert((x + size) <  opened_image.shape[0])
    assert((y + size) < opened_image.shape[1])

    # using pillow .crop function to sub-patch within the image
    # crops a SIZExSIZE chunk from the given image
    cropped_image = pillow_opened_image.crop((x,y,x+size,y+size)) 
    cropped_image = np.array(cropped_image)[:,:,0]

    # return
    return cropped_image





    
