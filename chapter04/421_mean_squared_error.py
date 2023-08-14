import numpy as np
from keras.datasets import mnist
from PIL import Image
import pickle

# 画像を表示する関数
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
def get_data(normalize=True,flatten=True,one_hot_label=False):
    (x_train, t_train), (x_test, t_test)= mnist.load_data()
    if normalize==True:
         x_train = x_train/255.0
         x_test  = x_test/255.0
    if flatten==True:
        x_train = x_train.reshape((x_train.shape[0],-1))
        x_test = x_test.reshape((x_test.shape[0],-1))
    return x_train,t_train

def init_network():
    with open("c:\\Users\\j12sh\\PROGRAM\\ZeroStartDeepLearning\\chapter03\\sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exp_x     = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y         = exp_x/sum_exp_x
    return y

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

x_train,t_train = get_data()

network = init_network()
W1 = network["W1"]
b1 = network["b1"]
W2 = network["W2"]
b2 = network["b2"]
W3 = network["W3"]
b3 = network["b3"]

x = x_train[0]
print(x.shape)
label = t_train[0]
print("label:",label)

a1 = np.dot(x,W1) + b1
z1 = sigmoid(a1)
a2 = np.dot(z1,W2) + b2
z2 = sigmoid(a2)
a3 = np.dot(z2,W3) + b3
z3 = sigmoid(a3)
y  = softmax(z3)

print(y)
t = [0,0,0,0,0,1,0,0,0,0]

ans = mean_squared_error(y,t)
print(ans)

