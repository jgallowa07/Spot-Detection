'''
generate training files in retinanet format:
path/to/image.jpg,x1,y1,x2,y2,class_name
'''

import cv2
import numpy as np

class_name = "synapse"
# input_dir = "/Users/nick/Desktop/BGMP/Term_project/bgmp-group-project-ml_neuron_id/dots/"
# input_path = input_dir + "L1-D02-z.bmp"

IMAGE_BASE = "L1-D0"
CURRENT_FP = None
CURRENT_IMG = ""

box_size = 16
box_adj = box_size/4


def format_output(x,y):
    top_coord = (int(x-box_adj),int(y-box_adj)); bot_coord = (int(x+box_adj),int(y+box_adj))
    cv2.rectangle(img,top_coord,bot_coord,(255,0,0),0)
    # print(CURRENT_IMG + "," + str(top_coord[0])+","+str(top_coord[1])+","+str(bot_coord[0])+","+str(bot_coord[1])+","+class_name)
    
    
    CURRENT_FP.write(CURRENT_IMG + "," + str(top_coord[0]) + "," + str(top_coord[1]) + "," + str(bot_coord[0]) + "," + str(bot_coord[1]) + "," + class_name + '\n' )

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),1,(0,0,255),-1)
        format_output(x,y)



for number in ['1','2']:
    for letter in ['g','s','z']:
        current_output = IMAGE_BASE + number + '-' + letter + '_output.csv'
        current_image = IMAGE_BASE + number + '-' + letter + '.bmp'

        CURRENT_IMG = current_image
        CURRENT_FP = open('annotation_output/' + current_output, 'w')

        img = cv2.imread(current_image)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img',draw_circle)

        while True:
            cv2.imshow('img',img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        print("test")
        cv2.destroyAllWindows()
        CURRENT_FP.close()
