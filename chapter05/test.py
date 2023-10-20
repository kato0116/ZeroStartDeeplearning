from PIL import Image
import numpy as np
from keras.datasets import mnist, cifar10
# データの取得
def get_data2(normalize=True,flatten=True,one_hot_label=True,num_class=10):
    (x_train, t_train), (x_test, t_test) = cifar10.load_data()
    if normalize:
        x_train = x_train/255
        x_test  = x_test/255
    if flatten:
        x_train = x_train.reshape(x_train.shape[0],-1)
        x_test  = x_test.reshape(x_test.shape[0],-1)
    # if one_hot_label:
    #     t_train = np.eye(num_class,dtype=int)[t_train]
    #     t_test  = np.eye(num_class,dtype=int)[t_test]
    if one_hot_label:
        t_train = np.eye(num_class, dtype=int)[t_train.flatten()]  # ワンホットエンコーディング
        t_test = np.eye(num_class, dtype=int)[t_test.flatten()] 
    return (x_train, t_train), (x_test, t_test)

    
(x_train,t_train),(x_test,t_test) = get_data2()   

print(x_train.shape)
print(t_train.shape) 
print(x_test.shape) 
print(t_test.shape)  
label = t_test[0]
print(label)
