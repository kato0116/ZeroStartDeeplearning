import numpy as np
from collections import OrderedDict
from .common.layers import *
from .common.optimizer import *


class SimpleConvNet:
    '''
    input_dim:(サンプル数、H,W)
    np.random.randn(フィルターの数, サンプル数, filter_size, filter_size)
    
    '''
    def __init__(self,input_dim=(1,28,28),
                 conv_param={"filter_num":50,"filter_size":5,
                             "pad":0,"stride":1},
                 hidden_size=100,output_size=10,weight_init_std=0.01):
        filter_num    = conv_param["filter_num"]
        filter_size   = conv_param["filter_size"]
        filter_pad    = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size    = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        # レイヤの生成
        self.layers = OrderedDict()
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = ReLU()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = AffineLayer(self.params['W2'], self.params['b2'])
        self.layers['Relu2']   = ReLU()
        self.layers['Affine2'] = AffineLayer(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()
        
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
            return x
        
    def loss(self,x,t):
        y = self.predict(x)
        return self.last_layer(y,t)
    
    # 勾配の計算、学習
    def gradient(self,x,t):
        # 順伝播
        Loss = self.loss(x,t)
        # 逆伝播
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values) # .values()でそのまま中身を実行 => 辞書をもう一つ作成
        layers.reverse()                    # layersを反転
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads["W1"]=self.layers["Conv1"].dW
        grads["b1"]=self.layers["Conv1"].db
        for idx in range(1,3):
            grads["W"+str(idx)] = self.layers["Affine"+str(idx)].dW
            grads["b"+str(idx)] = self.layers["Affine"+str(idx)].db
        return grads
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]


(x_train,t_train),(x_test,t_test) = get_data(flatten=False)  # mnist対応
x = x_train[0]
print(x.shape)

# network = SimpleConvNet(input_dim=(1,28,28), 
#                         conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
#                         hidden_size=100, output_size=10, weight_init_std=0.01)
# optimizer = AdaGrad()



# iters_num  = 2000 # 繰り返し回数
# train_size = x_train.shape[0]
# batch_size = 1000

# train_loss_list = [] # 学習データでの損失関数の値
# train_acc_list  = [] # 学習データでの認識精度の値
# test_acc_list   = [] # テストデータでの認識精度の値
# iter_per_epoch  = max(train_size/batch_size,1) # 1エポック当たりの繰り返し回数

# for i in range(iters_num):
#     batch_mask = np.random.choice(train_size,batch_size,replace=False)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
#     # 誤差逆伝播法で勾配の計算
#     grads = network.gradient(x_batch,t_batch)
#     # 更新
#     optimizer.update(network.params,grads)
#     loss = network.loss(x_batch,t_batch)
#     train_loss_list.append(loss)
#     if i%iter_per_epoch==0:
#         train_acc = network.accuracy(x_train,t_train)
#         test_acc  = network.accuracy(x_test,t_test)
#         train_acc_list.append(train_acc)
#         test_acc_list.append(test_acc)
#         # 経過表示
#         print(f'[更新数]{i:>4} [損失関数の値]{loss:.4f} '
#               f'[訓練データの認識精度]{train_acc:.4f} [テストデータの認識精度]{test_acc:.4f}')

# # 損失関数の値の推移を描画
# x = np.arange(len(train_loss_list))
# plt.plot(x, train_loss_list, label='loss')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.show()

# # 訓練データとテストデータの認識精度の推移を描画
# x2 = np.arange(len(train_acc_list))
# plt.plot(x2, train_acc_list, label='train acc')
# plt.plot(x2, test_acc_list, label='test acc', linestyle='--')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.xlim(left=0)
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()