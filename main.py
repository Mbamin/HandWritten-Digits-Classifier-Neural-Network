import sys 
sys.path.append('Scripts')
sys.path.append('Data')
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize
from ForwardPropagate import forwardPropagate
from BackPropagation import backprop
from SplitData import splitData
from DisplayData import displayData

print ('Loading Data')
print ('\n')
data = loadmat('Data/Data.mat')
#X is a 5000 * 400 matrix containing training 5000, 20x20 training examples that have been manipulated to into single rows 400 elements long
#Y is a 
X = data['X']
y = data['y']

print('Displaying 100 Random Images')

rand_indices = np.random.permutation(range(X.shape[0]))
sel = X[rand_indices[0:100], :]
displayData(sel)

print ('Seperating Data into Test and Training Sets')
print('\n')
#create Test and Train examples
X_test, X_train, y_train, Y_test, Y = splitData(X, y)
print ('One Hot Encoding Labels')
print ('\n')
encoder = OneHotEncoder(sparse=False,categories='auto')
y_onehot = encoder.fit_transform(y)
y_train = encoder.fit_transform(y_train)

print('Setting up Neural Network')
print ('\n')

# initial setup
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = .9

print('Initializing Parameters')
print ('\n')
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

print('Optimizing Theta Vectors using BackPropagation (This May take a While) ')
print ('\n')
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X_train, y_train, learning_rate), 
                method='TNC', jac=True, options={'maxiter': 300})

theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

print('Using Learned Parameters to Predict Numbers')
print ('\n')

a1, z2, a2, z3, h = forwardPropagate(X_train, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, Y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('Training Set Accuracy = {0} %'.format(accuracy * 100))
print ('\n')

a1, z2, a2, z3, h = forwardPropagate(X_test, theta1, theta2)
test_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(test_pred, Y_test)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('Test Set Accuracy = {0} %'.format(accuracy * 100))