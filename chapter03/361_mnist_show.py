from keras.datasets import mnist
import numpy as np
from PIL import Image

# 画像を表示する関数
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
(x_train, t_train),(x_test,t_test) = mnist.load_data()
# サイズを変形
x_train = x_train.reshape((x_train.shape[0],-1))
x_test  = x_test.reshape((x_test.shape[0],-1))

print(x_train.shape) # 60000,784
print(t_train.shape) # 60000,
print(x_test.shape)  # 10000,784
print(t_test.shape)  # 10000,

img = x_test[9986]
label = t_test[9986]
print(label)

img = img.reshape(28,28)
img_show(img)