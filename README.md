# Synapse detection.

This repository contains the code for UO BGMP synapse detection group working 
for the Washburn Lab (TODO LINK THIS) to create, and quantify the accuracy of a
synapse detection pipeline given confocal micrascopy images.

This project has three main aspects of the project
that are reflected in this repository. Annotation of images, quantification 
of other methods, and a custom deep learning model (TBA).

## Usage

This repository has a number of dependencies which should be strait-forward 
to install with your favorite package manager.

* numpy 
* matplotlib.pyplot
* PIL (Do we actually need this?)
* skimage.draw
* cv2
* Tensorflow/Keras
* nose

FINISH.

Simply run 
`git clone https://github.com/2019-bgmp/bgmp-group-project-ml-neuron-id.git`
before creating a new conda environment (reccomended)
and installing the dependencies above.

Once installed run:

```bash
python3 -m nose tests
```

To ensure everything is running as it should.

## CODE.

The directory for this structure is as follows:

```bash
├── Clicking
│   ├── cv2_example.py
│   ├── helpers.py
│   └── sandbox.py
├── Data
│   ├── Annotation
│   ├── Empirical
│   ├── Prepped_Annotation
│   └── Prepped_Empirical
├── DataPrep
│   ├── PrepEmp.py
│   ├── dev_sandbox.py
│   └── helpers.py
├── DeepLearning
│   ├── networks.py
│   └── train.py
├── README.md
├── TODO.md
├── project_documents
│   ├── BGMP_ML\ Bib.pdf
│   ├── Definitions.md
│   ├── Questions_for_Sarah.txt
│   ├── Stednitz_bgmp_proposal.pdf
│   ├── email_drafts.txt
│   └── readings
└── tests
    ├── __init__.py
    └── test_data_prep_helpers.py
```

**Clicking**

First, we have created a script which
will accurately annotate images by simply clicking on region of inters. Using
cv2, the script writes out PASCAL-VOL style csv which gives a bounding box
for each annotated synapse which has been clicked. These images
are used both to quantify existing methods, as well as train and test our own.

TODO: Give example.

**DataPrep**

Next we have Data prep scripts which contain functions that convert annotations
from various sources such as PASCAL-VOL, JSON, and symquant output into
`pixelmaps`. These maps are simply binary, 2-dimensional `numpy ndarray`'s 
which have 1's labeling synapses and 0's for background. 
Given the medium of pixelmaps, we can do interesting things like 
compute co-localization and 

TODO: Give example 

**DeepLearning**

TODO

**tests**

This repository contains all nosetests which will be used to test ALL 
functions implimented in this repository. To run the unit tests simply 
type `python3 -m nose tests`

**Data**

The data repository is split into 4 sub-directories. 

* Empirical - Contains raw confocal images in `.bmp` format
* Annotation - Contains raw annotation files from all methods such as clicking,
synquant, and nueral network output.
* Prepped Empirical - This is for empirical images which may have been manipulated 
for neural network training input mostly.
* Prepped Annotation - This is for pixelmaps to be used as targets for network
training
