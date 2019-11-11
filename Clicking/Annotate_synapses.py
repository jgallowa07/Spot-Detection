#! /usr/bin/env python3
"""
File: Helpers.py
Authors Jared Galloway, Nick Wagner, Annie Wang
Date: 11/20/2019

This file contains all helpful python
functions for scripts included in synapse detection.
"""


# TODO : Finish commenting and finish thes function.
# TODO : UNDER CONSTRUCTION
# next we can impliment a 


##############################################################################

#import os
#import sys

#import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image
#from skimage.draw import circle
#import cv2


# TODO Argparse


import argparse
import cv2
 
 
def annotate(event, x, y, flags, param):
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        
        # place a circle in
        cv2.circle(image, (x,y), 1, (255,0,255), -1)
        top_coord = (int(x-box_adj),int(y-box_adj))
        bot_coord = (int(x+box_adj),int(y+box_adj))
        cv2.rectangle(image,top_coord,bot_coord,(255,0,0),0)
        
        print(param[0])
        meta = [param[0],top_coord[0],top_coord[1],bot_coord[0],bot_coord[1]]
        param[1].write(f"{','.join([str(bb) for bb in meta])}")
 
		# draw a rectangle around the region of interest
#		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
#		cv2.imshow("image", image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-o", "--out", required=True, help="Path to the annotation csv")
ap.add_argument("-bbox", "--bounding_box_size", type=int, 
    required = False,
    default = 4, 
    help="this is the distance between the box center and the closest edge \
    For example, bbox = 4 means you are making a 8x8 bounding box \
    default is 4")
args = vars(ap.parse_args())

#if(args[output] == None):
#    base=os.path.basename(filepath)
#    output = os.path.splitext(base)[0] + ".csv"
output_fp = open(args["out"],"a")
box_adj = args["bounding_box_size"]

 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", annotate, [args["image"],output_fp])
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()


"""

img = None


##############################################################################
# UNDER CONSTRUCTION


def annotate(event, x, y, flags, params):
    print("in annotate")
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f":x {x}, y: {y}")
         
        cv2.circle(img,(x,y),100,(255,0,255),-1)
        #top_coord = (int(x-box_adj),int(y-box_adj))
        #bot_coord = (int(x+box_adj),int(y+box_adj))
        #cv2.rectangle(img,top_coord,bot_coord,(255,0,0),0)
        #print(params[0])
        
        #params[0].write(f{})
        
        

##############################################################################

def annotate_by_clicking(filepath, boxsize=16, output=None):

    # By default, we will write a csv with the 
    # same filemae as image, but with csv exgtention and in the cwd. 
    if(output == None):
        base=os.path.basename(filepath)
        output = os.path.splitext(base)[0] + ".csv"
    #output_fp = open()
    
    #print(output)
    #sys.exit()
    
    

    img = cv2.imread(filepath)
    cv2.namedWindow('img')    
    cv2.setMouseCallback('img', annotate, [filepath,output])

    while True:
        cv2.imshow('img',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
    #.close()

"""








