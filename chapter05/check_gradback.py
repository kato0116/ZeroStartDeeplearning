import numpy as np
from two_layer_net import *

## 勾配の確認
(x_train,t_train),(x_test,t_test) = get_data()
network = TwoLayer(input_size=784,hidden_size=50,output_size=10)
x_batch = x_train[:3]
t_batch = t_train[:3]
grad_numerical = network.numerical_gradient(x_batch,t_batch)
grad_backprop  = network.gradient(x_batch,t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+":"+str(diff))
