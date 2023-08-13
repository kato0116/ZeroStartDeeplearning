import numpy as np
from keras.datasets import mnist
from PIL import Image
import pickle

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# normalizeは正規化,flattenは1次元配列にするかどうか,one_hot_labelは正解となるラベルを1として格納する
def get_data(normalize=True,flatten=True,one_hot_label=False):
    (x_train, t_train), (x_test, t_test)= mnist.load_data()
    if normalize==True:
         x_train = x_train/255.0
         x_test  = x_test/255.0
    if flatten==True:
        x_train = x_train.reshape((x_train.shape[0],-1))
        x_test = x_test.reshape((x_test.shape[0],-1))
    return x_test,t_test

def init_network():
    with open("c:\\Users\\j12sh\\PROGRAM\\ZeroStartDeepLearning\\ch03\\sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ソフトマックス関数
def softmax(x):
    exp_x     = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y         = exp_x/sum_exp_x
    return y

def predict(network, x):
    W1,W2,W3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y  = softmax(a3)
    return y
    
x, t = get_data()
network = init_network()
accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p==t[i]:
        accuracy_cnt += 1
print(x.shape, t.shape)
print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
