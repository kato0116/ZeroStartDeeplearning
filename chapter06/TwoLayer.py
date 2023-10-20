import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10
from collections import OrderedDict
from optimizer import *

np.random.seed(13)

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

# 2層ニューラルネットワーク
class TwoLayer:
    def __init__(self,input_size,hidden_size,output_size,weight = 0.01):
        self.params = {} # 重みを保存する辞書
         
        self.params["W1"] = weight*np.random.randn(input_size,hidden_size) # ~(行数、列数)
        self.params["W2"] = weight*np.random.randn(hidden_size,output_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)
        
        # レイヤの作成
        self.layers = OrderedDict() # 追加した順番が保持される辞書
        # self.layers["Affine1"]  = AffineLayer(W=self.params["W1"],b=self.params["b1"])
        # self.layers["Sigmoid1"] = SigmoidLayer()
        # self.layers["Affine2"]  = AffineLayer(W=self.params["W2"],b=self.params["b2"])
        self.layers["Affine1"]  = AffineLayer(W=self.params["W1"],b=self.params["b1"])
        self.layers["ReLU1"]    = ReLULayer()
        self.layers["Affine2"]  = AffineLayer(W=self.params["W2"],b=self.params["b2"])
        self.LastLayer          = SoftmaxWithLoss()
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
        grads["W1"] = numerical_gradient(loss_W,self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W,self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W,self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W,self.params["b2"])
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
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db  
        return grads
            
# (x_train,t_train),(x_test,t_test) = get_data2() # cifar10対応
(x_train,t_train),(x_test,t_test) = get_data()  # mnist対応
network = TwoLayer(input_size=x_train.shape[1],hidden_size=50,output_size=10)
optimizer = AdaGrad()

iters_num  = 10000 # 繰り返し回数
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = [] # 学習データでの損失関数の値
train_acc_list  = [] # 学習データでの認識精度の値
test_acc_list   = [] # テストデータでの認識精度の値
iter_per_epoch  = max(train_size/batch_size,1) # 1エポック当たりの繰り返し回数

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size,replace=False)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 誤差逆伝播法で勾配の計算
    grads = network.gradient(x_batch,t_batch)
    # 更新
    optimizer.update(network.params,grads)
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    if i%iter_per_epoch==0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc  = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # 経過表示
        print(f'[更新数]{i:>4} [損失関数の値]{loss:.4f} '
              f'[訓練データの認識精度]{train_acc:.4f} [テストデータの認識精度]{test_acc:.4f}')

# 損失関数の値の推移を描画
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='loss')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()

# 訓練データとテストデータの認識精度の推移を描画
x2 = np.arange(len(train_acc_list))
plt.plot(x2, train_acc_list, label='train acc')
plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.xlim(left=0)
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()