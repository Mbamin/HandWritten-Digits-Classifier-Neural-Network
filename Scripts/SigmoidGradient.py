from sigmoid import sigmoid
import numpy as np

def sigmoidGradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))