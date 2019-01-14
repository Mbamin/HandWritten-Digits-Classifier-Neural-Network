import numpy as np
import matplotlib as plt
import warnings
warnings.filterwarnings("ignore")

def arrayToImage(i,x):
    X = np.reshape(x[i],(20,20)).T
    return X

def visualizePredictions(X_test,test_pred):
    rand_indices = np.random.permutation(range(X_test.shape[0]))
    sel = rand_indices[0:10]
    
    for i in sel:
        fig = plt.pyplot.figure()
        plt.pyplot.axis('off')
        ax = plt.pyplot.axes()
        im = ax.imshow(arrayToImage(i,X_test), cmap = 'gray')
        prediction = ax.text(.5, 1, '', color= 'white')
        pred = test_pred[i]
        #because I have mapped the digit 0 to the 10th index 
        if (pred == 10):
            pred = 0
        prediction.set_text('Prediction: %d'% (pred))
        input("Press Enter to continue...")
        plt.pyplot.show()

        