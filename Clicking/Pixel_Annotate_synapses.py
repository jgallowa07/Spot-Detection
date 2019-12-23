#! /usr/bin/env python3
"""
File: 
Authors Jared Galloway
Date: 11/20/2019
"""

import argparse
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-o", "--out", required=True, help="Path to the annotation csv")
args = vars(ap.parse_args())

# a unique set of clicked points
clicked_points = set()

# load the numpy image, clone it, and setup the mouse callback function
img = np.load(args["image"], allow_pickle = True)

# initialize an empty pixelmap for output
pixelmap = np.zeros(img.shape[:-1]).astype(np.int)
 
def annotate(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and (x,y) not in clicked_points:
        clicked_points.add((x,y))
        cv2.circle(image, (x,y), 0, (255,0,255), -1)
        pixelmap[x,y] = 1

#output_fp = open(args["out"],"a")

cv2.imwrite('color_img.jpg', img)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow("image", img);
image = cv2.imread("color_img.jpg")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", annotate, [args["image"],pixelmap])
 
# keep looping until the <esc> key is pressed
# display the image and wait for a keypress
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
np.transpose(pixelmap).dump(args["out"])



