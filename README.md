# Super Mega McEasy Dot Net. 

Obviously we need a better name. haha. 

**Overview**
This repository contains the code for UO BGMP synapse detection group working 
for the 
[Washburn Lab](https://ion.uoregon.edu/content/philip-washbourne) 
to create, and quantify the accuracy of a
synapse detection pipeline given confocal microscopy images.
Generally we use train-through-simulation technique which requires 
_no_ annotated empirical images, while remaining supervised.
The simulations parameters can be tuned simply to more closely match data of interest.
While this has been done before, (TODO cite) our method focuses on simplicity, effeciency. 
colocalization between image channels, and easy training of a model which can then be
used directly on empirical data.

**1. simulation** :
This package takes a few parameters such as dot radius, background noise variance,
and a few others to simulate the _colocalization_ of 2D exponential  bumps across
3 different channels. Often with florescent microscopy images such immuno-labeling,
it is necessary for "dots" to be seen across some channels or all channels to be 
of interest. Our simulation interface allows users to specify the amount 
of complete colocalization (a bump shared across all channels) bi-localized? 
(across two channels, for any combination), and singlet bumps on any one of the
channels independently. These bumps are centered at some x,y location on 
any one channel, and the activation of each pixel depends on the euclidean 
distance to the center of the dot given, where the number of pixels for 
any one center is defined by the radius in pixels. Given distance to the 
center of a dot, the activation is calculated by an exponential distribution,
and incremented by some some Gaussian noise. -- TODO finish

**2. Deep Learning**
Once a dataset has modeled dots/bumps/spots sufficiently, 
we use a relatively strait-forward convolutional neural network architecture
which takes 3-channel images (3D tensors), and outputs probability maps 
(2D tensors) describing whether or not that pixel is part of a dot-of-interest.
We train the neural network to identify _colocalization_ between layers - meaning

We measure the accuracy of our model by the F1-score (precision and recall values)
described below. -- TODO finish


## Installation

This repository has a number of dependencies which should be strait-forward 
to install with your favorite package manager.

* numpy 
* matplotlib.pyplot
* PIL (Do we actually need this?)
* skimage.draw
* cv2
* Tensorflow/Keras
* nose
* read-roi

FINISH.

Simply run 
`git clone https://github.com/2019-bgmp/bgmp-group-project-ml-neuron-id.git`
before creating a new conda environment (recommended)
and installing the dependencies above.

Once installed run:

```bash
python3 -m nose tests
```

To ensure everything is running as it should.

## Quickstart

Here is an example of how to simulate a single 32 X 32 image with 
1 r+g+b dot, 1 r+b, 1 r+g, 1 g+b, 1 r, 1 g, 1b ... For each image 
generate we also generate the 2D pixel map describing the groud truth
for bumps which colocalize at least `coloc_thresh` times across the RGB
channels.  

The driver function included with `fmi_simulator` is `simulate`, with 
the following description:

```python
help(fmi_simulator.simulate)
```

```
Help on function simulate in module fmi_simulator:

simulate(
    num_samples=10, 
    width=32, 
    height=32, 
    coloc_thresh=1, 
    coloc_n=[1, 1, 1, 1, 1, 1, 1], 
    coloc_p=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
    radii_n=3, 
    radii_p=0.85, 
    spot_noise=0.2, 
    point_noise=0.2, 
    background_noise=0.2)

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
    
    :param: radii_n, radii_p <int> - The radii distribution of of dots simulated. n and p
        represent the binomial distribution of radii of all simulated dots.
    
    :param: spot_noise <float [0-1]> the variance of the guassian noise added to all dots.
    
    # TODO Not sure this is necessary
    :param: point_noise <float [0-1]> the variance of the guassian noise subtracted from
         the center of all dots.
    
    :param: background_noise <float [0-1]> the variance of the guassian noise added to the
        background.
```

```python
>>> # simulate example
>>> import fmi_simulator
>>> x, y = fmi_simulator.simulate()
>>> x.shape
(10, 32, 32, 3)
>>> y.shape
(10, 32, 32, 1)


>>> # visualize 
>>> plt.imshow(x[0])
>>> plt.show()
```


## Clicking

First, we have created a script which
will accurately annotate images by simply clicking on region of inters. Using
cv2, the script writes out PASCAL-VOL style csv which gives a bounding box
for each annotated synapse which has been clicked. These images
are used both to quantify existing methods, as well as test our own.

TODO: We should make a script which can will allow the user to make a pixel map 
of a 32 X 32 image! This would be a much better way to quantify, no?

## DataPrep

Conceptually, Dataprep is just a plethora of helpful function we use 
to simulate, train on, and transform data to accomplish the task of 
spot detection.

