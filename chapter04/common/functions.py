import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from PIL import Image

def get_data(normalize=True,flatten=True,one_hot_label=True,num_class=10):
    (x_train, t_train), (x_test, t_test)= mnist.load_data()
    if normalize==True:
         x_train = x_train/255.0
         x_test  = x_test/255.0
    if flatten==True:
        x_train = x_train.reshape((x_train.shape[0],-1))
        x_test = x_test.reshape((x_test.shape[0],-1))
    if one_hot_label==True:
        t_train = np.eye(num_class,dtype=int)[t_train]  # One-hotエンコーディングを適用
        t_test = np.eye(num_class,dtype=int)[t_test]    # One-hotエンコーディングを適用
        
    return (x_train, t_train), (x_test, t_test)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exp_x     = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y         = exp_x/sum_exp_x
    return y

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))