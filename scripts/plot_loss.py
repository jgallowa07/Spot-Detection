from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0,"../")
from DataPrep.helpers  import *





network = "control"
loss = np.load(f"./output/loss_{network}.out",allow_pickle = True)
val_loss = np.load(f"./output/val_loss_{network}.out", allow_pickle = True)



fig, ax = plt.subplots()
ax.plot(loss[2:])
ax.plot(val_loss[2:])
ax.set_yscale('log')
ax.set_ylabel("Mean Squared Error")
plt.show()
#

#x = np.load(f"./output/x_{network}.out", allow_pickle = True)

#tensor_to_3dmap(x[0][:,:,0], out = None)
