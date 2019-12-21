from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0,"../../")
from DataPrep.helpers  import *

s = 0.1 
p = 0.45
b = 0.2
l = 2

#filename = f"{l}l_s{int(s*100)}_p{int(p*100)}_b{int(b*100)}"
filename = f"l2_s05_p45_b20"

pred = np.load(f"../output/pred_{filename}.out",allow_pickle=True)
y = np.load(f"../output/y_{filename}.out",allow_pickle=True)
x = np.load(f"../output/x_{filename}.out",allow_pickle=True)

for i in range(5):
    plt.imshow(x[i])
    plt.show()
    plt.imshow(np.squeeze(y[i]))
    plt.show()
    plt.imshow(np.squeeze(pred[i]))
    plt.show()
    print(np.sum(pred[i]))
    print(np.sum(y[i]))
    
    
    #print(f1_score_pixel_v_prob(prediction = np.squeeze(pred[i]), target = np.squeeze(y[i]), threshold = 0.95))
print(f1_score_pixel_v_prob(prediction = np.squeeze(pred), target = np.squeeze(y), threshold = 0.95))
