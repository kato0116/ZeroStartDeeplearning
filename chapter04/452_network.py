import numpy as np
import matplotlib.pyplot as plt
from common.functions import *

class TwoLayer():
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.grams = {} # 重みとバイアスの辞書作成
        self.grams["W1"] = weight_init_std*np.random.rand(input_size,hidden_size)
        self.grams["b1"] = np.zeros(hidden_size)
        self.grams["W2"] = weight_init_std*np.random.rand(hidden_size,output_size)
        self.grams["b2"] = np.zeros(output_size)    

    def predict(self,x):
        W1,W2 = self.grams["W1"],self.grams["W2"]
        b1,b2 = self.grams["b1"],self.grams["b2"]   
        
        a1 = np.dot(x,W1)  + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y  = softmax(a2)
        return y
    
    def loss(self,x,t):
        y = self.predict(x)
        E = cross_entropy_error(y,t)
        return E
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        
        accuracy = np.sum(y==t)/float(x.shape(0))
        return accuracy

    # paramsは重み・バイアスの辞書、gramsは勾配の辞書
    def numerical_gradient(self,x,t):
        loss_W = lambda:self.loss(x,t) # def loss_W(): return self.loss(x,t)と同義
        grads  = {} # 勾配を保存する辞書
        grads["W1"] = numerical_gradient(loss_W,self.grams["W1"])
        grads["b1"] = numerical_gradient(loss_W,self.grams["b1"])
        grads["W2"] = numerical_gradient(loss_W,self.grams["W2"])
        grads["b2"] = numerical_gradient(loss_W,self.grams["b2"])       
        
        return grads

(x_train, t_train), (x_test, t_test) = get_data()

network = TwoLayer(input_size=784,hidden_size=50,output_size=10)

iters_num = 1000             # 繰り返し回数
train_size = x_train.shape[0] # 学習するデータの数
batch_size = 10               # バッチサイズ
learning_rate = 0.1           # 学習率

train_loss_list = []          # 損失関数の結果を保存するリスト
train_acc_list  = []
test_acc_list   = []

for i in range(1):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    print(batch_mask)
    x_batch = x_train[batch_mask]
    
    t_batch = t_train[batch_mask]
    # 勾配の計算
    grad = network.numerical_gradient(x_batch,t_batch)
    
    # パラメータの更新
    for key in ("W1","b1","W2","b2"):
        network.grams[key] -= learning_rate*grad[key]
    
    # 学習経過の記録
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
    

# エポックごとの損失をプロット
plt.plot(train_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
