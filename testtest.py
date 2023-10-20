import numpy as np
from collections import OrderedDict
from chapter05.two_layer_net import *

(x_train,t_train),(x_test,t_test) = get_data()
train_size = x_train.shape[0]
batch_size = 10
print(x_train.shape)
batch_mask = np.random.choice(train_size,batch_size,replace=False)
print(batch_mask.shape)
x_train = x_train[batch_mask]
print(x_train.shape)
