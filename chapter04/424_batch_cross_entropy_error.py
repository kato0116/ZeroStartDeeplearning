import numpy as np
from keras.datasets import mnist
from PIL import Image
import pickle

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

# バッチ対応版の交差エントロピー誤差関数
## one_hot表現の場合
def cross_entropy_error(y,t):
    if y.ndim==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    delta = 1e+7
    return -np.sum(t*np.log(y+delta))/batch_size

## ラベルの場合
def cross_entropy_error_label(y,t):
    if y.dim==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e+7))/batch_size


(x_train, t_train), (x_test, t_test) = get_data()
print(x_train.shape)
print(t_train.shape)

train_size = x_train[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

