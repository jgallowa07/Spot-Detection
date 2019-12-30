
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
import cv2
from read_roi import read_roi_zip

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

    image_pixelmap: numpy 2d array corresponding to the pixelmap to be sub-patched
    size: the SIZExSIZE chunk to be grabbed
    height: tuple specifying the y start and stop positions (start, stop)  
            DEFAULT: (256,1024)
    width: tuple specifying the x start and stop positions (start, stop)  
            DEFAULT: (256,768)
    
    This function will return a numpy ndarray of SIZExSIZE 2-dimentional binary numpy ndarrays
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

    list_of_paths: list of strings corresponding to the paths of the images wanting to be
                sub-imaged
    size: the SIZExSIZE chunk to be grabbed
    height: tuple specifying the y start and stop positions (start, stop)  
            DEFAULT: (256,1024)
    width: tuple specifying the x start and stop positions (start, stop)  
            DEFAULT: (256,768)

    This function will return a list of numpy ndarrays, of which contain all of the sub-images
    for that given empirical image. There will be one item in the list for each empirical 
    image given
    """

    sub_empirical = []
    for num in range(len(list_of_paths)):
        pillow_opened_image = Image.open(list_of_paths[num])
        temp_sub_images = []
        for i in range(height[0],height[1],size):  # this for loop isolates only the region of the image specified in the parameters
            for j in range(width[0],width[1],size):
                temp_pic = pillow_opened_image.crop((i,j,i+size,j+size)) # grabbing SIZExSIZE chunks and storing them in an array
                # 7
                temp_pic = np.array(temp_pic)[:,:,0]
                temp_sub_images.append(temp_pic)
        temp_sub_images = np.array(temp_sub_images)
        sub_empirical.append(temp_sub_images)
    
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


    pixelmap1, pixelmap2: two pixelmaps of the same size which we would like
    to evaluate and get a metric of how they compare

    This function will return a float value between 0 and 1. 0 being the worst
    the model could be doing amd 1 being the best.
    """
    assert(pixelmap1.shape == pixelmap2.shape)
    assert(pixelmap1.dtype == np.int and pixelmap2.dtype == np.int)

    true_positive = np.sum(np.bitwise_and(pixelmap1, pixelmap2))
    false_positive = np.sum(np.bitwise_and(pixelmap1, ~pixelmap2))
    false_negative = np.sum(np.bitwise_and(~pixelmap1, pixelmap2))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return 2/((1/precision) + (1/recall))

###

# TODO Impliment
def generate_whole_dataset_stub():
    """
    Come up with a simpler interface for running and saving a bunch of 
    these images.
    """
    pass

###

# TODO This could obviously be made much more complex
def add_normal_noise_to_image(image, gaussian_bg_sd, background_only = True):
    """
    this image adds background noise (absolute value 
    Gaussian centered at zero) with variance gaussian_bg_sd,
    to each pixel channel which is currently
    not already activated (activation 0). 

    for more noise, simply add variance.
    """
    # add background noise to an image
    gaussian_bg_mean = 0
    bg = np.abs(np.random.normal(gaussian_bg_mean, gaussian_bg_sd, image.shape))

    # add the noise to image param, skipping the dots, if required
    # image[image == 0] += bg[image == 0] if backgound_only else image += bg
    if background_only:
        image[image == 0] += bg[image == 0] 
    else:
        image += bg

    # correct for values above 1!
    image[image > 1] = 1

    

###

def generate_simulated_microscopy_sample(
        colocalization = [5] + [0 for _ in range(6)],
        width = 32,
        height = 32,
        radius = 2,
        coloc_thresh = 3,
        s_noise = 0.2,
        p_noise = 0.2,
        ):
    
    # TODO max radius size? make a radius vector for each layer of
    # x,y coordinates to introduce some more noise! mo betta.

    # TODO Clean up and re-do docs and testing!



    """
    :param: colocalization <list> - a list which contains the 7 colocal counts!
        the params should be in the following order:
            idx - colocal meaning
            0 - all_layers share. as well as the pixelmap
            1 - just the 0, and 1 share
            2 - just the 1 and 2 share
            3 - just the 0 and 2 share
            4 - just 0
            5 - just 1
            6 - just 2
            
    :param: width <int> - width of the sample 
    
    :param: height <int> - height of the sample
    
    :param: radius <int> - radius of bumps

    :return: (3D numpy tensor, 2D numpy tensor) - this is going to 
        be the simulated 

    Here, we take in amount of spots wanted as either colocalized
    on any combination of channels 0, 1, and 2, or singlet on any layer.

    This leaves 7 possibilities:
        
        1X Complete Co-localization - all layers have this bump
        3X double co-loc, 3 choose 2 combinations of co-pair-bumps
        3X singlet bumps


    Given these 7 params, this function computes the x,y vectors
    for each three layers so they may be created seperately and 
    finally stacked into the 3D tensor representing simulated 
    confocal image with parameterized co-localization. 
    """
    
    assert(len(colocalization) == 7)
    assert(radius < (width // 2) and radius < (height // 2))
    assert(coloc_thresh in [1,2,3])
    
    # initialize out empty layers.
    #layer0, layer1, layer2, pixelmap = ([] for _ in range(4))

    # Hm, if you bored, you could generalize this 
    # colocalization algorithm getting all combinations in a set
    
    # the first three are the layers for the simulated sample,
    # the last later is the pixelmap target
    layers_list = [[] for _ in range(4)]
    combs = [[0,1,2],[0,1],[1,2],[0,2],[0],[1],[2]]
  
    for i,layers in enumerate(combs):
        for num_dots in range(colocalization[i]):
            x = np.random.randint(radius, width - radius)
            y = np.random.randint(radius, height - radius)
            for layer_index in layers:
                layers_list[layer_index] += [(x,y)]
            if len(layers) >= coloc_thresh:
                layers_list[3] += [(x,y)]

    channels = [simulate_single_layer(
        layers_list[i], width, height, radius,
        s_noise = s_noise, p_noise = p_noise) for i in range(3)]
    simulated_sample = np.stack(channels,axis=2)    
    pixelmap_target = simulate_single_layer(
        layers_list[3], width, height, radius, is_pixelmap = True)

    return simulated_sample, pixelmap_target

###

def simulate_single_layer(
        xy_list,
        width,
        height,
        radius,
        is_pixelmap = False,
        s_noise = 0.2,
        p_noise = 0.2
        ):
    """
    This function will simulate a single layer given the coordinates for each 
    exponential bump! This function is utilized by the generate_simulated_microscopy_sample
    function so that it can generate multiple layers and combine them to create 
    the desired example image, or just a pixelmap for the simulated target.

    :param: xy_list <list> - list that contains sets of x and y locations (x,y)
        that will the locations for the bumps (synapses) in this particular
        layer

    :param: width <int> - width of the sample 
    
    :param: height <int> - height of the sample
    
    :param: radius <int> - radius of bumps
    
    :param: is_pixelmap <boolean> - boolean to specify whether or not the layer
        being created is going to be one of three layers for an example image or
        a pixelmap representing the target for the simulation

    :param: s_noise <float> - 

    :param: p_noise <float> - 

    
    :return: sim_bump <ndarray> - numpy array representing one fully simulated
        layer.
    """    
    

    # not implimented yet
    assert(type(radius) == int)

    # init the tensor to be returned.
    sim_bump = np.zeros([width, height])

    # Step through all the x,y locations where the dots will be located 
    # on each channel,
    for x,y in xy_list:
    
        # Draw nice circle and init an array to store
        # respective activations,
        xx,yy = circle(x,y,radius)
        if is_pixelmap:
            sim_bump[xx,yy] = 1
            continue

        activation_list = np.zeros(len(xx))

        # for each location that is a synapse
        # we are going to compute the activation 
        for i in range(len(xx)):

            # use pythagorian theorem to compute radius on discrete space!
            diff_from_center = math.sqrt((xx[i] - x)**2 + (yy[i] - y)**2)

            # This is where we sample from the exponential "bump"
            # Question, How dow we make this bump wider, @ Annie
            # I would like for the majority of the numbers not to 
            # be so small :)
            # TODO: Did we ever get this resolved?
            activation = np.exp(-((0.5*diff_from_center)**2))
       
            # we then add gaussian noise the add another level of randomness 
            activation_list[i] = activation + np.abs(np.random.normal(0,s_noise))
            #print("s_noise",s_noise)
            #assert(s_noise == 0.1)
            #activation_list[i] = activation + np.abs(np.random.normal(0,0.1))

        # finally, population the tensor.
        sim_bump[xx,yy] += activation_list

    # Okay here, lets correct for the number things greater and equal to one.
    # the main idea is: a by-product of our algorithm is that the center of all
    # synapses have an activation == to 1. we should correct for this 
    # because it's not realistic
    # TODO This could be done slightly more effeciently.
    if not is_pixelmap:
        sim_bump[sim_bump > 1] = 1
        num_ones = len(sim_bump[sim_bump == 1])
        sim_bump[sim_bump == 1] += -1 * np.abs(np.random.normal(0,p_noise,num_ones))
    
    assert(len(sim_bump[sim_bump > 1]) == 0)

    return sim_bump

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

#def tensor_to_3dmap(tensor, out = None, cmap = "bone"):
#    """
#    A function which takes in a 2D numpy array and produces 
#    a heatmap.
#
#    if a filename is given to out then it will save the fig,
#    otherwise it will attempt to open the png with matplotlib.
#    """
#
#    X = np.arange(0, tensor.shape[0])
#    Y = np.arange(0, tensor.shape[1])
#    X, Y = np.meshgrid(X, Y)
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    surf = ax.plot_surface(X, Y, tensor, rstride=1, 
#        cstride=1, cmap=cmap, linewidth=0, antialiased=False)
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    if out == None:
#        plt.show()
#    else:
#        plt.savefig(out)
#
#    return None
#
###

def simple_simulator(num_samples, width, height, 
        coloc_thresh, colocalization, radius = 2,
        s_noise = 0.2,
        p_noise = 0.2,
        b_noise = 0.2):
    """
    This function allows for a very non-complex way to generate a 
    dataset of simulated example and target images.

    :param: num_samples <int> - number of samples to generate
            
    :param: width <int> - width of the sample
    
    :param: height <int> - height of the sample

    :param: coloc_thresh <int> - One of [1,2,3], which is the number of 
        images with the same dot needed to colocalize to the final x and y
    
    :param: colocalization <list> - a list which contains the 7 colocal counts.
        the params should be in the following order:
            idx - colocal meaning
            0 - all_layers share. as well as the pixelmap
            1 - just the 0, and 1 share
            2 - just the 1 and 2 share
            3 - just the 0 and 2 share
            4 - just 0
            5 - just 1
            6 - just 2

    :param: radius <int> - radius of bumps

    :param: s_noise <float> - 

    :param: p_noise <float> -

    :param: b_noise <float> -

    :return: [x:4D numpy tensor, y:4D numpy tensor] - 
            x: simulated sample images  
            y: simulated target pixelmaps
    """

    # initialize x and y as numpy arrays of all zeros to be the placeholders of 
    # images to be simulated
    x = np.zeros([num_samples, width, height, 3])
    y = np.zeros([num_samples, width, height])

    # loop through and call the generate_simulated_microscopy_sample function
    # the amount of times as the desired number of samples. The same parameters
    # are used for each call
    for i in range(num_samples):
        X, Y = generate_simulated_microscopy_sample(
            colocalization = colocalization,
            width = width,
            height = height,
            radius = radius,
            coloc_thresh = coloc_thresh,
            s_noise = s_noise,
            p_noise = p_noise)

        # add normal noise to the example image based on the level of
        # b_noise desired
        add_normal_noise_to_image(X,b_noise)
        
        # add each x and y to our already initialized placeholder lists
        x[i] = X
        y[i] = Y

    # reshape the final target list to be 4 dimensional
    y = np.reshape(y, [num_samples, width, height, 1])
    
    return x, y

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




    
