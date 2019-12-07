from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0,"../")
from DataPrep.helpers  import *

pred = np.load("output/pred_2l_35n.out",allow_pickle=True)
y = np.load("output/test_y_2l_35n.out",allow_pickle=True)
x = np.load("output/test_x_2l_35n.out",allow_pickle=True)

#for i in range(10):
    #plt.imshow(x[i])
    #plt.show()
    #plt.imshow(np.squeeze(y[i]))
    #plt.show()
    #plt.imshow(np.squeeze(pred[i]))
    #plt.show()
    #print(np.sum(pred[i]))
    #print(np.sum(y[i]))
    
    
    #print(f1_score_pixel_v_prob(prediction = np.squeeze(pred[i]), target = np.squeeze(y[i]), threshold = 0.95))
print(f1_score_pixel_v_prob(prediction = np.squeeze(pred), target = np.squeeze(y), threshold = 0.95))
