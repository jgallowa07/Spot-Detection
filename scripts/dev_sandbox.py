
from helpers import *
from matplotlib import pyplot as plt
import numpy as np
import sys



# note, filepaths are relative to where you run nose.
width, height = 1024, 1024
anno_file = sys.argv[1]

# make sure the images are the same size        
pixelmap = dot_click_annoation_file_to_pixelmap(
    anno_file = anno_file,
    width = width,
    height = height,
    dot_radius = 2)

plt.imshow(pixelmap)
plt.show()
