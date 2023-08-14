import numpy as np
from keras.datasets import mnist
from PIL import Image
import pickle

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

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
E = cross_entropy_error(np.array(y),np.array(t))

print(E)