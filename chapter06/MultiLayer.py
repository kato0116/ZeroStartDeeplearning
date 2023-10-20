import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10
from collections import OrderedDict
from optimizer import *

# データの取得
def get_data(normalize=True,flatten=True,one_hot_label=True,num_class=10):
    (x_train, t_train), (x_test, t_test)= mnist.load_data()
    if normalize:
        x_train = x_train/255
        x_test  = x_test/255
    if flatten:
        x_train = x_train.reshape(x_train.shape[0],-1)
        x_test  = x_test.reshape(x_test.shape[0],-1)
    if one_hot_label:
        t_train = np.eye(num_class,dtype=int)[t_train]
        t_test  = np.eye(num_class,dtype=int)[t_test]
    return (x_train, t_train), (x_test, t_test)

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



# softmax関数
def softmax(x):
    c = np.max(x, axis=-1, keepdims=True)
    exp_a = np.exp(x - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

# 交差エントロピーバッチ対応
def cross_entropy_error(y,t):
    if y.ndim==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size) 
    
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # np.nditerで多次元配列の要素を列挙
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:

        idx = it.multi_index  # it.multi_indexは列挙中の要素番号
        tmp_val = x[idx]  # 元の値を保存

        # f(x + h)の算出
        x[idx] = tmp_val + h
        fxh1 = f()

        # f(x - h)の算出
        x[idx] = tmp_val - h
        fxh2 = f()

        # 勾配を算出
        grad[idx] = (fxh1 - fxh2) / (2 * h)
    
        x[idx] = tmp_val  # 値を戻す
        it.iternext()

    return grad

# ReLUレイヤ
class ReLULayer:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
# Sigmoidレイヤ
class SigmoidLayer:
    def __init__(self):
        self.out = None
    def forward(self,x):
        self.out = 1/(1+np.exp(-x))
        return self.out
    def backward(self,dout):
        dx = dout*self.out*(1-self.out)
        return dx
    
# Affineレイヤ
class AffineLayer:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.X = None
        self.db = None
        self.dW = None
    def forward(self,X):
        self.X = X
        out = np.dot(X,self.W)+self.b
        return out
    def backward(self,dout):
        dX      = np.dot(dout,self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout,axis=0) # データの列ごとの総和
        return dX

# class BatchNormalization:
#     def __init__(self):
#         self.mu    = None
#         self.sigma = None
#     def forward(self,x):
        
        

# Softmaxと交差エントロピーのレイヤ
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)* (dout / batch_size)
        return dx

# 5層ニューラルネットワーク
class MultiLayer:
    def __init__(self,input_size,hidden_size_list,output_size):
        self.params = {} # 重みを保存する辞書
        self.all_layer_list = [input_size]+hidden_size_list+[output_size]
        
        self.__init_weight__()
        self.__init_layer__()
    
    # ReLUなのでHeの初期値を採用（Sigmoidの場合はXavierの初期値推奨）
    def __init_weight__(self):
        for idx in range(1,len(self.all_layer_list)):
            # weight_std = 0.01
            weight_std = np.sqrt(2/self.all_layer_list[idx-1]) # Heの初期値
            self.params["W"+str(idx)] = weight_std*np.random.randn(self.all_layer_list[idx-1],self.all_layer_list[idx])
            self.params["b"+str(idx)] = np.zeros(self.all_layer_list[idx])
    
    # レイヤの初期化
    def __init_layer__(self):
        self.layers = OrderedDict() # 追加した順番が保持される辞書
        for idx in range(1,len(self.all_layer_list)-1):
            self.layers["Affine"+str(idx)]    = AffineLayer(W=self.params["W"+str(idx)],b=self.params["b"+str(idx)])
            self.layers["ReLU"  +str(idx)]    = ReLULayer()
        idx = len(self.all_layer_list)-1
        self.layers["Affine"+str(idx)]  = AffineLayer(W=self.params["W"+str(idx)],b=self.params["b"+str(idx)])
        self.LastLayer                  = SoftmaxWithLoss()
        
    # 順伝播
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x) # 各層でxを更新していく
        return x
    # 損失関数(交差エントロピー)
    def loss(self,x,t):
        y = self.predict(x)
        return self.LastLayer.forward(y,t)
    # 認識精度の計算
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        if t.ndim != 1:t = np.argmax(t,axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy
    # 数値微分
    def numerical_gradient(self,x,t):
        loss_W = lambda:self.loss(x,t)
        grads = {}
        for idx in range(1,len(self.all_layer_list)):
            grads["W"+str(idx)] = numerical_gradient(loss_W,self.params["W"+str(idx)])
            grads["b"+str(idx)] = numerical_gradient(loss_W,self.params["b"+str(idx)])
        return grads
    
    # 逆伝播
    def gradient(self,x,t):

        # forward
        Loss = self.loss(x,t)
        # backward
        dout = 1
        dout = self.LastLayer.backward(dout)
        layers = list(self.layers.values()) # .values()でそのまま中身を実行 => 辞書をもう一つ作成
        layers.reverse()                    # layersを反転
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1,len(self.all_layer_list)):
            grads["W"+str(idx)] = self.layers["Affine"+str(idx)].dW
            grads["b"+str(idx)] = self.layers["Affine"+str(idx)].db
        return grads