import numpy as np
from MultiLayer import *
from optimizer import *

(x_train,t_train),(x_test,t_test) = get_data()  # mnist対応
network = MultiLayer(input_size=x_train.shape[1],hidden_size_list=[100,100,100],output_size=10)

optimizer = AdaGrad()



iters_num  = 2000 # 繰り返し回数
train_size = x_train.shape[0]
batch_size = 1000

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