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

clicked_points = set()
 
def annotate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and (x,y) not in clicked_points:
        clicked_points.add((x,y))
        cv2.circle(image, (x,y), 0, (255,0,255), -1)
        param[1].write(f"{x},{y}\n")

output_fp = open(args["out"],"a")

# load the image, clone it, and setup the mouse callback function
img = np.load(args["image"], allow_pickle = True)
cv2.imwrite('color_img.jpg', img)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow("image", img);
image = cv2.imread("color_img.jpg")
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


