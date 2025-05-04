import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
## NOTE : this code is from the tme3 of MAPSI of last semester ! 

def get_train_data():
    data = pkl.load(open("usps.pkl",'rb')) 

    X_train = np.array(data["X_train"],dtype=float)
    

    Y_train = data["Y_train"]
    

    X_train = X_train[:-1]
    Y_train = Y_train[:-1]

    return X_train,Y_train,9 #batch number

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


 
 