import numpy as np
from keras.datasets import mnist
from common.functions import *

class TwoLayerNet():
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params = {} # 重みの用の辞書作成
        self.params["W1"] = weight_init_std *\
                            np.random.rand(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std *\
                            np.random.rand(hidden_size,output_size)
        self.params["b2"] = np.zeros(output_size)
    
    def predict(self,x):
        W1,W2 = self.params["W1"],self.params["W2"]
        b1,b2 = self.params["b1"],self.params["b2"]
        
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y  = softmax(a2)
        
        return y
    
    def loss(self,x,t):
        y = self.predict(x)          # 入力データから出力を計算
        E = cross_entropy_error(y,t) # 交差エントロピー誤差を計算
        return E
    
    def accuracy(self,x,t):
        y = self.predic(x)
        print(y)
        print(t)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        
        accuracy = np.sum(y==t)/float(x.shape(0))
        return accuracy
    
    # x:入力データ,t:教師データ
    def numerical_gradient(self,x,t):
        loss_W = lambda W:self.loss(x,t)
        
        grads = {}
        grads["W1"] = numerical_gradient(loss_W,self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W,self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W,self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W,self.params["b2"])
        return grads
(x_train, t_train), (x_test, t_test) = get_data(one_hot_label=False)
net = TwoLayerNet(input_size=784,hidden_size=100,output_size=10)
print(net.params["W1"].shape)
print(net.params["b1"].shape)
print(net.params["W2"].shape)
print(net.params["b2"].shape)

y = net.predict(x_train[0])
print(y[5])