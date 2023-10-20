import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000個のデータ
node_num = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size = 5  # 隠れ層が5層
activations = {}  # ここにアクティベーションの結果を格納する

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 初期値の値をいろいろ変えて実験しよう！
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # Xavierの初期値  <==sigmoid関数と相性〇
    # 標準偏差1/sqrt(n)の重み　n:前層のノード数
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # Heの初期値  <==ReLU関数と相性〇
    # 標準偏差2/sqrt(n)の重み　n:前層のノード数
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a = np.dot(x, w)


    # 活性化関数の種類も変えて実験しよう！
    # z = sigmoid(a)
    z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# activations = init_weight_Xavier(x,node_num,hidden_layer_size,activations)
for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+"-layer")
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(),30,range=(0,1))
    
plt.show()
        