# python sub_image.py -g L1-D01-g.bmp -s L1-D01-s.bmp -z L1-D01-z.bmp

from PIL import Image
# import matplotlib.image as plt
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="A program to convert clicked annotation into an image")
    parser.add_argument("-g", "--gfile", help="use to specify the g file name", required=True, type = str)
    parser.add_argument("-s", "--sfile", help="use to specify the s file name", required=True, type = str)
    parser.add_argument("-z", "--zfile", help="use to specify the z file name", required=True, type = str)
    return parser.parse_args()

args = get_args()               # calls get_args method from above assigns the arguments to args
G_FILE = args.gfile          # assigning g image file path as string to global varible
S_FILE = args.sfile          # assigning s image file path as string to global varible
Z_FILE = args.zfile          # assigning z image file path as string to global varible

SUB_IMAGES = []

G_IMG = Image.open(G_FILE)
G_IMG = G_IMG.convert('L')
G_NUMPY = np.asarray(G_IMG.getdata()).reshape(G_IMG.size)

S_IMG = Image.open(S_FILE)
S_IMG = S_IMG.convert('L')
S_NUMPY = np.asarray(S_IMG.getdata()).reshape(S_IMG.size)

Z_IMG = Image.open(Z_FILE)
Z_IMG = Z_IMG.convert('L')
Z_NUMPY = np.asarray(Z_IMG.getdata()).reshape(Z_IMG.size)

# print(G_NUMPY.shape, S_NUMPY.shape, Z_NUMPY.shape)

count = 0
g_sub_images = []
s_sub_images = []
z_sub_images = []
for i in range(256,1024,32):
    for j in range(256,768,32):
        g_temp_array = G_NUMPY[i:i+32,j:j+32]
        g_sub_images.append(g_temp_array)

        s_temp_array =S_NUMPY[i:i+32,j:j+32]
        s_sub_images.append(s_temp_array)

        z_temp_array = Z_NUMPY[i:i+32,j:j+32]
        z_sub_images.append(z_temp_array)
        count += 1


SUB_IMAGES = np.array([g_sub_images, s_sub_images, z_sub_images])

print("Number of sub-images created:", count)
print("Shape of X", SUB_IMAGES.shape)