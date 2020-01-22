# Spot Detection Using Convolutional Neural Networks 

## Abstract

Spot detection is an important task in many fields such as Biology, Astronomy, and Physics. Unfortunately, the task of counting spots in images can take experts many laborious hours to do by hand. Recent studies (cite) have shown that Machine Learning has potential to automate this task. However, It remains a problem to acquire pixel-level annotations needed as targets for any type of supervised learning.  Here, we present a novel method of training convolutional neural networks on simulated images --- allowing users to completely side-step the need for human annotated data sets. Our results exemplify that this method offers a competitive F1 score on empirical, fluorescent microscopy images when compared to other supervised machine learning methods.

To see a full walk through of our experimental workflow, it is reccomended you look at the Jupyter Notebook found in `Notebooks/train_spot.ipynb`

## Simulation

Often the natural word follows explicit rules and patterns which we can closely model using known probability distributions and heuristic patterns. When looking at empirical images containing spots of activation, the clear pattern is a focal point of activation which clearly tapers off in the surrounding pixels in a round shape, with exceptions. In our simulations, we modeled this with a 2-dimensional Gaussian ``bump'' scaled by the radius for any given spot. Concretely, given 
($x_{f}, y_{f}$) describing the focal point for a bump, the activation of any pixel ($x_{p}, y_{p}$) within a dot of radius $r$, and distance to focal point


## Our Model


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

For a full walkthrough of our experimental workflow, it is reccomended you run our
Jupyter Notebook `train_spot.ipynb` found in the `notebooks` directory. 
Tweak the parameters to your desire, however be aware that the deep learning
can both be quite computationally expensive.

If you would like to use the helpful simulator we've created, 
you can simply import the the script `scipts/dot_simulator.py`
and use either of two functions descibed below.

`dot_simulator.single_layers_simulator()`

```python
>>> import scripts.dot_simulator as dot
>>> help(dot.single_layers_simulator)
```

```
Help on function simulate_single_layers in module scripts.dot_simulator:

simulate_single_layers(num_samples=10, 
                        width=32, 
                        height=32, 
                        num_dots_n=5, 
                        num_dots_p=0.5, 
                        radii_n=3, 
                        radii_p=0.85, 
                        spot_noise=0.2, 
                        point_noise=0.2, 
                        background_noise=0.2)
    This function will generate single layer 
    samples of simulated flurescent microscopy images
    along with the respective ground truth annotation. 
    The simulations themselves are defined by:
    
    :param: width, height <int> - The width and height of the simulated images
        defined in number of pixels.
    
    :param: coloc_thresh <int> - This is a number, either 1, 2, or 3 which 
        which primarily affects how we annotate the images as being true positives
        or negative examples. Concretely, this is the threshold of number of layers
        which any dot must colocalize in order to be considered as a synapse.
    
    :param: num_dots_n, num_dots_p <int> - these parameters define the expected 
        binomial distribution of dots among all simulated images where n is the number
        of trials (bernoulli trial for dot or no dot) and p is the probability of success.
        The default is n = 5, and p = 0.5 meaning, on average, we expect 2.5 dots per image.
        
    :param: radii_n, radii_p <int> - The radii distribution of of dots simulated. n and p
        represent the binomial distribution of radii of all simulated dots. NOTE that 
        we floor the radius size of any one dot to be 2.
    
    :param: spot_noise <float [0-1]> the variance of the guassian noise added to all dots.
    
    # TODO Not sure this is necessary
    :param: point_noise <float [0-1]> the variance of the guassian noise subtracted from
         the center of all dots.
    
    :param: background_noise <float [0-1]> the variance of the guassian noise added to the
        background.
```

Or, if you would like to simulate three channels of an image with colocalization patterns,
you can use our function 

`dot_simulator.colocalized_triplets_simulator()`

```python3
>>> help(dot.colocalized_triplets_simulator)
```

```
Help on function colocalized_triplets_simulator in module scripts.dot_simulator:

colocalized_triplets_simulator(num_samples=10, width=32, height=32, coloc_thresh=1, coloc_n=[1, 1, 1, 1, 1, 1, 1], coloc_p=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], radii_n=3, radii_p=0.85, spot_noise=0.2, point_noise=0.2, background_noise=0.2)
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

## Clicking

We have created two scripts which can be used for annotation.
It is reccomended that if you would like to test how well simulations
have performed on your empirical data, that you do a pixel-level annotation
for a patch greater than or equal to 64 X 64, (be sure to )

