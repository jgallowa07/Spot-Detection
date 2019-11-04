
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import circle

def get_args():
    parser = argparse.ArgumentParser(description="A program to convert clicked annotation into an image")
    parser.add_argument("-i", "--input", help="use to specify the input file name", required=True, type = str)
    return parser.parse_args()

args = get_args()               # calls get_args method from above assigns the arguments to args
input_data = args.input          # assigning forward read file name as string to global varible


IMAGE = np.zeros((1024,1024), dtype=np.uint8)
COORDINATES = []
XY = []


with open(input_data, 'r') as inFile:
    for line in inFile:
        line = line.strip().split(',')
        curr_coordinates = [line[1], line[2], line[3], line[4]]
        COORDINATES.append(curr_coordinates)

for i in range(len(COORDINATES)):
    x = int(COORDINATES[i][0]) + 4
    y = int(COORDINATES[i][1]) + 4

    rr, cc = circle(x, y, 2)
    IMAGE[rr,cc] = 1
    XY.append([x,y])

plt.imshow(IMAGE, cmap="gray")
plt.show()
