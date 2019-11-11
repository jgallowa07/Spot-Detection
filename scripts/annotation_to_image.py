# python annotation_to_image.py -g annotation_output/L1-D01-g_output.csv -s annotation_output/L1-D01-s_output.csv -z annotation_output/L1-D01-z_output.csv

import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import circle
from skimage.measure import label, regionprops

def get_args():
    parser = argparse.ArgumentParser(description="A program to convert clicked annotation into an image")
    parser.add_argument("-g", "--gfile", help="use to specify the g file path", required=True, type = str)
    parser.add_argument("-s", "--sfile", help="use to specify the s file path", required=True, type = str)
    parser.add_argument("-z", "--zfile", help="use to specify the z file path", required=True, type = str)
    parser.add_argument("-r", "--radius", help="use to specify the desired radius to extend pixel by", default = 2, type = int)
    return parser.parse_args()

args = get_args()               # calls get_args method from above assigns the arguments to args
G_FILE = args.gfile          # assigning g annotation file name as string to global varible
S_FILE = args.sfile          # assigning s annotation file name as string to global varible
Z_FILE = args.zfile          # assigning z annotation file name as string to global varible
RADIUS = args.radius






IMAGE1 = np.zeros((1024,1024), dtype=np.uint8)
IMAGE2 = np.zeros((1024,1024), dtype=np.uint8)
IMAGE3 = np.zeros((1024,1024), dtype=np.uint8)
COLOCALIZED = np.zeros((1024,1024), dtype=np.uint8)
COORDINATES1 = []
COORDINATES2 = []
COORDINATES3 = []
XY1 = []
XY2 = []
XY3 = []

SUB_IMAGES = []  # global list to hold all of the 32x32 sub images



with open(G_FILE, 'r') as gfile, open(S_FILE, 'r') as sfile, open(Z_FILE, 'r') as zfile:
    for line in gfile:
        line = line.strip().split(',')
        curr_coordinates = [line[1], line[2], line[3], line[4]]
        COORDINATES1.append(curr_coordinates)
    for line in sfile:
        line = line.strip().split(',')
        curr_coordinates = [line[1], line[2], line[3], line[4]]
        COORDINATES2.append(curr_coordinates)
    for line in zfile:
        line = line.strip().split(',')
        curr_coordinates = [line[1], line[2], line[3], line[4]]
        COORDINATES3.append(curr_coordinates)

for i in range(len(COORDINATES1)):
    x = int(COORDINATES1[i][0]) + 4
    y = int(COORDINATES1[i][1]) + 4

    rr, cc = circle(x, y, RADIUS)

    count = 0
    for i in range(len(rr)):
        if (rr[i-count] >= 1024 or rr[i-count] < 0):
            rr = np.delete(rr, i-count)
            cc = np.delete(cc, i-count)
            count += 1

    IMAGE1[rr,cc] = 1
    XY1.append([x,y])

for i in range(len(COORDINATES2)):
    x = int(COORDINATES2[i][0]) + 4
    y = int(COORDINATES2[i][1]) + 4

    rr, cc = circle(x, y, RADIUS)

    count = 0
    for i in range(len(rr)):
        if (rr[i-count] >= 1024 or rr[i-count] < 0):
            rr = np.delete(rr, i-count)
            cc = np.delete(cc, i-count)
            count += 1

    IMAGE2[rr,cc] = 1
    XY2.append([x,y])

for i in range(len(COORDINATES3)):
    x = int(COORDINATES3[i][0]) + 4
    y = int(COORDINATES3[i][1]) + 4

    rr, cc = circle(x, y, RADIUS)
    r = []
    c = []
    count = 0
    for i in range(len(rr)):
        if (rr[i-count] >= 1024 or rr[i-count] < 0):
            rr = np.delete(rr, i-count)
            cc = np.delete(cc, i-count)
            count += 1
    
    IMAGE3[rr,cc] = 1
    XY3.append([x,y])

# props = regionprops(IMAGE1)
# print(props[0].centroid)
# print('test')

# plt.imshow(IMAGE1, cmap="gray")
# plt.show()

# plt.imshow(IMAGE2, cmap="gray")
# plt.show()

# plt.imshow(IMAGE3, cmap="gray")
# plt.show()


COLOCALIZED1 = np.bitwise_and(IMAGE1, IMAGE2)
COLOCALIZED2 = np.bitwise_and(IMAGE1, IMAGE3)
COLOCALIZED3 = np.bitwise_and(IMAGE2, IMAGE3)

COLOCALIZED = np.bitwise_or(COLOCALIZED1, COLOCALIZED2)
COLOCALIZED = np.bitwise_or(COLOCALIZED, COLOCALIZED3)

count = 0
for i in range(256,1024,32):  # this for loop isolates only the region of the image we care about
    for j in range(256,768,32):
        temp_array = COLOCALIZED[i:i+32,j:j+32] # grabbing 32x32 chunks and storing them in an array
        SUB_IMAGES.append(temp_array)
        count += 1

SUB_IMAGES = np.array(SUB_IMAGES)

print("Number of sub-images created:", count)
print("Shape of Y:", SUB_IMAGES.shape)



# plt.imshow(COLOCALIZED, cmap='gray')
# plt.show()
