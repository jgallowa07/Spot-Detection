
"""
File: Helpers.py
Authors Jared Galloway, Nick Wagner, Annie Wang
Date: 11/20/2019

This file contains all helpful python
functions for scripts included in synapse detection.
"""


# TODO : Finish commenting and finish thes function.
# next we can impliment a 


##############################################################################

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import circle
import cv2


# TODO Argparse












##############################################################################
# UNDER CONSTRUCTION


def annotate(event, x, y, flags, params):
    """

    """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),1,(0,0,255),-1)
        cv2.rectangle(img,top_coord,bot_coord,(255,0,0),0)
        top_coord = (int(x-box_adj),int(y-box_adj)) 
        bot_coord = (int(x+box_adj),int(y+box_adj))      
        
        

##############################################################################

def annotate_by_clicking(filepath, boxsize=16, output=None)
    """

    """

    # By default, we will write a csv with the 
    # same filepath as image itself. 
    if(output == None):
        base=os.path.basename(filepath)
        output = os.path.splitext(base)[0] + ".csv"
    
    print(output)
    sys.exit()
    

    img = cv2.imread(current_image)
    cv2.namedWindow('img')
    
    cv2.setMouseCallback('img',draw_circle, [output])

    while True:
        cv2.imshow('img',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
    CURRENT_FP.close()










