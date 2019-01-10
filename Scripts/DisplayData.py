import numpy as np
import matplotlib.pyplot as plt

def displayData(X):
    
    num_plots = int(np.size(X,0)**.5)
    fig, ax = plt.subplots(num_plots,num_plots,sharex=True,sharey=True)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix transposes
            img = X[img_num,:].reshape(20,20).T
            ax[i][j].imshow(img,cmap='gray')
            ax[i][j].axis('off')
            img_num += 1
    plt.show()

    return (fig, ax)