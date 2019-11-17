import os
import sys
sys.path.insert(0,"../")

import numpy as np

from DataPrep.helpers import *
import numpy as np
import matplotlib.pyplot as plt

syn_file1='../Data/Annotation/synquant_output/z=4/RoiSet_g.zip'
syn_file2='../Data/Annotation/synquant_output/z=4/RoiSet_s.zip'
syn_file3='../Data/Annotation/synquant_output/z=4/RoiSet_z.zip'
pixelmap11=synquant_to_pixelmap_stub(syn_file1)
pixelmap22=synquant_to_pixelmap_stub(syn_file2)
pixelmap33=synquant_to_pixelmap_stub(syn_file3)
synquant_colocalization_map = colocaliztion([pixelmap11,pixelmap22,pixelmap33])

# img = plt.imshow(synquant_colocalization_map)
# plt.show(img)

width, height = 1024, 1024
anno_file1 = "../Data/Annotation/annotation_output/L1-D01-g_output.csv"
anno_file2 = "../Data/Annotation/annotation_output/L1-D01-s_output.csv"
anno_file3 = "../Data/Annotation/annotation_output/L1-D01-z_output.csv"

# Create three separate pixelmaps to use for colocalization testing       
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


# test colocalization with three pixelmaps
phil_output = colocaliztion([pixelmap1,pixelmap2,pixelmap3])

# img = plt.imshow(phil_output)
# plt.show(img)



# print(synquant_colocalization_map.shape)
# print(phil_output.shape)
# self.assertEqual(synquant_colocalization_map.shape,(1024,1024))
# self.assertEqual(phil_output.shape,(1024,1024))
f1_output = f1_score(pixelmap11, pixelmap1)

print(f1_output)