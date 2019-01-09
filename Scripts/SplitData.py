import numpy as np

def splitData(X, y):
    
    X_train = np.zeros((1,400))
    y_train = np.zeros((1,1))
    Y = np.zeros((4000,1))
    Y_test = np.zeros((1000,1))
    X_test = np.zeros((1,400))
    
    rand_perm = np.random.permutation(X.shape[0])  
    
    for i in range (5000):
            
            if (i < 4000):
                X_train = np.vstack((X_train,X[rand_perm[i]]))
                y_train = np.vstack((y_train,y[rand_perm[i]]))
                Y[i] = y[rand_perm[i]]
            else:
                X_test = np.vstack((X_test,X[rand_perm[i]]))
                Y_test[i-4000] = y[rand_perm[i]] 
                
    X_train = np.delete(X_train,(0),axis = 0)
    y_train = np.delete(y_train,(0),axis = 0)
    X_test  = np.delete(X_test,(0), axis = 0)
    
    return X_test, X_train, y_train, Y_test, Y