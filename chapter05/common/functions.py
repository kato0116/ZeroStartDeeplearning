import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from PIL import Image

def get_data(normalize=True,flatten=True,one_hot_label=True,num_class=10):
    (x_train, t_train), (x_test, t_test)= mnist.load_data()
    # 正規化
    if normalize==True:
         x_train = x_train/255.0
         x_test  = x_test/255.0
    # 入力画像を1次元に表現
    if flatten==True:
        x_train = x_train.reshape((x_train.shape[0],-1))
        x_test = x_test.reshape((x_test.shape[0],-1))
    # 0～9のラベルを[0,0,....,1]の0,1で表現
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
## one_hot表現の場合
def cross_entropy_error(y,t):
    
    if y.ndim==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
        
    
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


# 勾配の計算(2変数に対応)
def numerical_gradient_2d(f, x):
    h = 1e-4
    print(x.size)
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

# 勾配の計算(多次元に対応)
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x,flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index # 現在のインデックスを表示
        tmp_val = x[idx] # 現在のxの値をコピー
        
        # f(x+h)
        x[idx] = tmp_val + h
        fxh1   = f()
        # f(x-h)
        x[idx] = tmp_val - h
        fxh2   = f()
        
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx]    = tmp_val
        it.iternext() # 次の要素に移動
    return grad 

# lrは学習率, step_numは試行回数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x.copy()

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad  # xの更新
        plt.scatter(x[0],x[1])
    return x