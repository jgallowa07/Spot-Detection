"""
Simulator: A script containing the source code needed for 
    generating simulated, flurecent immuno-labeled microscopy images. 

Authors: Jared G. Nick W. Annie W.
"""

import os
import sys

import numpy as np
from helpers import *


def simulator(
        num_samples = 10,
        width = 32,
        height = 32,
        coloc_thresh = 1,
        coloc_n = [1 for _ in range(7)],
        coloc_p = [0.5 for _ in range(7)],
        radii_n = 3,
        radii_p = 0.85,
        spot_noise = 0.2,
        point_noise = 0.2,
        background_noise = 0.2
        ):

    """
    This function will generate samples of simulated flurescent microscopy images
    along with the respective ground truth annotation. The simulations themselves are
    defined by:

    :param: width, height <int> - The width and height of the simulated images
        defined in number of pixels.

    :param: coloc_thresh <int> - This is a number, either 1, 2, or 3 which 
        which primarily affects how we annotate the images as being true positives
        or negative examples. Concretely, this is the threshold of number of layers
        which any dot must colocalize in order to be considered as a synapse.

    :param: coloc_n, coloc_p <list[int]> - For each of the seven possible colocalization
        patterns (1 all layers + 3 combinations of two layers + 3 individual layers),
        the user passes the simulater a binomial distribution defined by n, p which 
        determines the number of each colocalization patters - of each type - we expect 
        among all our simulations. For example, the default is n = 1, and p = 0.5 for all
        7 colocalization patterns. This means that each pattern has a 50% of being in a 
        single simulated image. 

    # TODO This could be a distribution as well
    :param: radius <int> - The radius of the simulated dots. 

    :param: spot_noise <float [0-1]> the variance of the guassian noise added to all dots.

    # TODO Not sure this is necessary
    :param: point_noise <float [0-1]> the variance of the guassian noise subtracted from
         the center of all dots.

    :param: background_noise <float [0-1]> the variance of the guassian noise added to the
        background.

    """

    # some very basic error handling
    assert(type(num_samples) == int)
    assert(type(width) == int)
    assert(type(height) == int)
    assert(type(coloc_thresh) == int)
    assert(coloc_thresh <= 3 and coloc_thresh >= 1)
    assert(len(coloc_n) == 7)
    assert(len(coloc_p) == 7)

    x = np.zeros([num_samples, width, height, 3])
    y = np.zeros([num_samples, width, height])
    for i in range(num_samples):
        coloc = [np.random.binomial(n,p) for n,p in zip(coloc_n,coloc_p)]
        X, Y = generate_simulated_microscopy_sample(
            colocalization = coloc,
            radii_n = radii_n,
            radii_p = radii_p,
            width = width,
            height = height,
            coloc_thresh = coloc_thresh,
            s_noise = spot_noise,
            p_noise = point_noise)

        add_normal_noise_to_image(X,background_noise)
        
        x[i] = X
        y[i] = Y

    y = np.reshape(y, [num_samples, width, height, 1])
    
    return x, y


def simple_simulator(
        num_samples, 
        width, height, 
        coloc_thresh, 
        colocalization, 
        radius = 2,
        s_noise = 0.2,
        p_noise = 0.2,
        b_noise = 0.2
        ):

    """
    A very non-complex simulator
    """

    x = np.zeros([num_samples, width, height, 3])
    y = np.zeros([num_samples, width, height])
    for i in range(num_samples):
        X, Y = generate_simulated_microscopy_sample(
            colocalization = colocalization,
            width = width,
            height = height,
            radius = radius,
            coloc_thresh = coloc_thresh,
            s_noise = s_noise,
            p_noise = p_noise)

        add_normal_noise_to_image(X,b_noise)
        
        x[i] = X
        y[i] = Y

    y = np.reshape(y, [num_samples, width, height, 1])
    
    return x, y


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

    if background_only:
        image[image == 0] += bg[image == 0] 
    else:
        image += bg

    # correct for values above 1!
    image[image > 1] = 1

    return None


def generate_simulated_microscopy_sample(
        colocalization = [5] + [0 for _ in range(6)],
        radii_n = None,
        radii_p = None,
        width = 32,
        height = 32,
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
    #assert(radius < (width // 2) and radius < (height // 2))
    assert(coloc_thresh in [1,2,3])
    
    # initialize out empty layers.
    #layer0, layer1, layer2, pixelmap = ([] for _ in range(4))

    # Hm, if you bored, you could generalize this 
    # colocalization algorithm getting all combinations in a set
    
    # the first three are the layers for the simulated sample,
    # the last later is the pixelmap target
    layers_list = [[] for _ in range(4)]
    radii_list = [[] for _ in range(4)]
    combs = [[0,1,2],[0,1],[1,2],[0,2],[0],[1],[2]]
  
    for i,layers in enumerate(combs):
        for num_dots in range(colocalization[i]):
            radius = np.random.binomial(n = radii_n, p = radii_p)
            if radius == 0:
                continue
            x = np.random.randint(radius, width - radius)
            y = np.random.randint(radius, height - radius)
            for layer_index in layers:
                layers_list[layer_index] += [(x,y)]
                radii_list[layer_index] += [radius]
            if len(layers) >= coloc_thresh:
                layers_list[3] += [(x,y)]
                radii_list[3] += [radius]

    channels = [simulate_single_layer(
        layers_list[i], radii_list[i], width, height,
        s_noise = s_noise, p_noise = p_noise) for i in range(3)]
    simulated_sample = np.stack(channels,axis=2)    
    pixelmap_target = simulate_single_layer(
        layers_list[3], radii_list[3], width, height, is_pixelmap = True)

    return simulated_sample, pixelmap_target

# TODO
def simulate_single_pixelmap_stub(
        ):
    pass

def simulate_single_layer(
        xy_list,
        radii,
        width,
        height,
        is_pixelmap = False,
        s_noise = 0.2,
        p_noise = 0.2
        ):
    """
    This function will simulate a single layer 
    given the x,y coordinates for each 
    exponential bump.
    """    
    

    # not implimented yet
    assert(len(radii) == len(xy_list))

    # init the tensor to be returned.
    sim_bump = np.zeros([width, height])

    # Step through all the x,y locations where the dots will be located 
    # on each channel,
    for dot, (x,y) in enumerate(xy_list):
    
        # Draw nice circle and init an array to store
        # respective activations,

        radius = radii[dot]

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

            # This is how we link radius of dots to the activation.
            # we essentially just scale the exponential function 
            # dots with wider radius have a higher variance distribution.
            #c = np.log(5) / ((-1 * radius) ** 2)
            c = 2.302 / ((-1 * radius) ** 2)

            # This is where we sample from the exponential "bump"
            # Question, How dow we make this bump wider, raise the exponent.
            activation = np.exp(-1 * c * (diff_from_center**2))
           
            # we then add guassian noise the add another level of randomness 
            activation_list[i] = activation + np.abs(np.random.normal(0,s_noise))

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



