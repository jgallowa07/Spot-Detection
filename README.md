# Spot Detection Using Convolutional Neural Networks 

## Abstract

Spot detection is an important task in many fields such as Biology, Astronomy, and Physics. Unfortunately, the task of counting spots in images can take experts many laborious hours to do by hand. Recent studies (cite) have shown that Machine Learning has potential to automate this task. However, It remains a problem to acquire pixel-level annotations needed as targets for any type of supervised learning.  Here, we present a novel method of training convolutional neural networks on simulated images --- allowing users to completely side-step the need for human annotated data sets. Our results exemplify that this method offers a competitive F1 score on empirical, fluorescent microscopy images when compared to other supervised machine learning methods.

## Simulation

<a href="https://www.codecogs.com/eqnedit.php?latex=A()&space;=" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A()&space;=" title="A() =" /></a>

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

Here is an example of how to simulate a single 32 X 32 image with 
1 r+g+b dot, 1 r+b, 1 r+g, 1 g+b, 1 r, 1 g, 1b ... For each image 
generate we also generate the 2D pixel map describing the groud truth
for bumps which colocalize at least `coloc_thresh` times across the RGB
channels.  

The driver function included with `fmi_simulator` is `simulate`, with 
the following description:

```python
```

```

```


## Clicking

First, we have created a script which
will accurately annotate images by simply clicking on region of inters. Using
cv2, the script writes out PASCAL-VOL style csv which gives a bounding box
for each annotated synapse which has been clicked. These images
are used both to quantify existing methods, as well as test our own.


