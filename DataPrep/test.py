from helpers import *
#priInt(ll)

for i in range(5):
    x,y = generate_simulated_microscopy_sample(colocalization = [1,1,1,1,1,1,1], 
        width=32, height=32, coloc_thresh = 2)
    x = np.zeros([32,32,3])
    
    add_normal_noise_to_image(x,1.0)
    plt.imshow(x)
    plt.show()
    plt.imshow(y)
    plt.show()
