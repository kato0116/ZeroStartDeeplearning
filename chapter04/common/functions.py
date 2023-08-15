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

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# softmax関数
def softmax(x):
    exp_x     = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y         = exp_x/sum_exp_x
    return y

# 交差エントロピー誤差の計算
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

# 勾配の計算
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 勾配を0で初期化
    for i in range(x.size):
        tmp = x[i]
        # f(x+h)
        x[i] = tmp + h
        fxh1 = f(x)
        # f(x-h)
        x[i] = tmp - h
        fxh2 = f(x)

        # xを元に戻す
        x[i] = tmp
        # 勾配
        grad[i] = (fxh1 - fxh2) / (2 * h)  # 修正: '+' を '-' に変更
    return grad

# lrは学習率, step_numは試行回数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x.copy()

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad  # xの更新
        plt.scatter(x[0],x[1])
    return x