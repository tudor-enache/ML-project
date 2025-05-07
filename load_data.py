import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
## NOTE : this code is from the tme3 of MAPSI of last semester ! 

def get_train_data():
    data = pkl.load(open("usps.pkl",'rb')) 

    X_train = np.array(data["X_train"],dtype=float)
    maxi = np.max(X_train)
    mini = np.min(X_train)

    X_train = (X_train - mini)/(maxi- mini)

    Y_train = data["Y_train"]
    

    X_train = X_train[:-1]
    Y_train = Y_train[:-1]

    return X_train,Y_train,9 #batch number

def labels_to_one_hot(Y):
    one_hot = []
    for y in Y :
        h = np.zeros(10)
        h[y] = 1
        one_hot.append(h)
    return np.array(one_hot)

def get_test_data():
    data = pkl.load(open("usps.pkl",'rb')) 
    X_test = np.array(data["X_test"],dtype=float)
    Y_test = data["Y_test"]

    return X_test,Y_test

def show_digit(title,tab):
    plt.figure(figsize=(4,4))
    plt.imshow(tab.reshape(16,16),cmap="gray")
    plt.colorbar()
    plt.title(title)
    plt.show()


 
 