import numpy as np
from keras.datasets import mnist
from common.functions import *

class simpleNet():
    def __init__(self):
        self.W = np.random.rand(2,3) # ガウス分布で初期化
        
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        return loss

net = simpleNet()
print("重みパラメータ")
print(net.W)

x = np.array([0.6,0.9])
p = net.predict(x)
print(np.argmax(p))   # 最大値のインデックスを表示
t = np.array([0,0,1]) # 正解のラベル

E = net.loss(x,t)
print(E)