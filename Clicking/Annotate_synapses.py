#! /usr/bin/env python3
"""
File: Helpers.py
Authors Jared Galloway, Nick Wagner, Annie Wang
Date: 11/20/2019

This file contains the script to annotate a synapse image.
This will annotate one image specified by the path 
given to --image, with output witten to --out, and 
--bbox representing the size of the bounding box around 
clicking position. 
"""

import argparse
import cv2

##############################################################################
 
def annotate(event, x, y, flags, param):

    # to make clicking more accurate, for me the puts
    # the x,y coord EXACTLY where the point of my clicker is.
    x -= 1 if x >= 1 else x
    y -= 2 if x >= 2 else x

    if event == cv2.EVENT_LBUTTONDOWN:
        
        # place a circle in where we're clicking
        cv2.circle(image, (x,y), 1, (255,0,255), -1)
        top_coord = (int(x-box_adj),int(y-box_adj))
        bot_coord = (int(x+box_adj),int(y+box_adj))

        # Bounding box
        cv2.rectangle(image,top_coord,bot_coord,(255,0,0),0)
        meta = [param[0],top_coord[0],top_coord[1],bot_coord[0],bot_coord[1]]
        param[1].write(f"{','.join([str(bb) for bb in meta])}\n")
 
##############################################################################

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

##############################################################################

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

output_fp.close()
cv2.destroyAllWindows()


