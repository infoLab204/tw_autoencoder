import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import random 
import sys
import cv2

from scipy.io import loadmat

from matplotlib import pyplot

data_type=sys.argv[1]  ## select data type : MNIST, FMNIST, SVHN and CIFIR-10

if data_type=="MNIST" :
    (X_train_full, Y_train_full), (X_test, Y_test) = mnist.load_data() 
elif data_type=="FMNIST" :
    (X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()
elif data_type=="CIFAR":
    (X_train_full, Y_train_full), (X_test, Y_test) = cifar10.load_data()
elif data_type=="SVHN" :
    train_raw = loadmat('train_32x32.mat')
    test_raw = loadmat('test_32x32.mat')

    X_train_full=np.array(train_raw['X'])
    X_test=np.array(test_raw['X'])

    Y_train_full=train_raw['y']
    Y_test=test_raw['y']

    X_train_full=np.moveaxis(X_train_full, -1,0)
    X_test=np.moveaxis(X_test, -1,0)
else :
    print("data type error")

print(X_train_full.shape, Y_train_full.shape)
print(X_test.shape, Y_test.shape)

X=np.concatenate([X_train_full,X_test])
Y=np.concatenate([Y_train_full,Y_test])
if data_type=="SVHN" :
    X=X[:90000,:,:,:]
    Y=Y[:90000,:]


if data_type=="CIFAR" or data_type=="SVHN" :  ## BGR to Grayscale, Grayscale to binary scale
    X_train_sel=np.zeros((1,1024))
    for i in range(X.shape[0]) :
        img_gray = cv2.cvtColor(X[i,:,:,:], cv2.COLOR_BGR2GRAY)
        dst=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,3)
        dst=dst.reshape(1,1024).astype('float32')/255.0
        X_train_sel=np.vstack((X_train_sel,dst))

    X_train_sel=np.delete(X_train_sel, 0 , axis = 0)
    X=X_train_sel

if data_type == "MNIST" or data_type=="FMNIST" :
    X = X.reshape(70000, 784).astype('float32') / 255.0

if data_type == "CIFAR" :
    cv=1
    str_kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=50)
    for train_index, test_index in str_kf.split(X,Y) :
        X_train, X_test=X[train_index], X[test_index]
        Y_train, Y_test=Y[train_index], Y[test_index]
        X_test= pd.DataFrame(X_test)
        Y_test= pd.DataFrame(Y_test)
        X_test.to_csv(f"{data_type}_X_data_CV{cv}.csv", header=False, index=False)
        Y_test.to_csv(f"{data_type}_Y_label_CV{cv}.csv", header=False, index=False)
        cv=cv+1
elif data_type=="SVHN" :
    cv=1
    str_kf = StratifiedKFold(n_splits=9, shuffle=True, random_state=50)
    for train_index, test_index in str_kf.split(X,Y) :
        X_train, X_test=X[train_index], X[test_index]
        Y_train, Y_test=Y[train_index], Y[test_index]
        X_test= pd.DataFrame(X_test)
        Y_test= pd.DataFrame(Y_test)
        X_test.to_csv(f"{data_type}_X_data_CV{cv}.csv", header=False, index=False)
        Y_test.to_csv(f"{data_type}_Y_label_CV{cv}.csv", header=False, index=False)
        cv=cv+1
else :
    cv=1
    str_kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=50)
    for train_index, test_index in str_kf.split(X,Y) :
        X_train, X_test=X[train_index], X[test_index]
        Y_train, Y_test=Y[train_index], Y[test_index]
        X_test= pd.DataFrame(X_test)
        Y_test= pd.DataFrame(Y_test)
        X_test.to_csv(f"{data_type}_X_data_CV{cv}.csv",header=False, index=False)
        Y_test.to_csv(f"{data_type}_Y_label_CV{cv}.csv", header=False, index=False)
        cv=cv+1

## End of load_data.py
