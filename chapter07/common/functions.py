import numpy as np
from keras.datasets import mnist, cifar10

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

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (サンプル数, チャンネル, 高さ, 幅)
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    
    Returns
    -------
    col : 2次元配列
    """

    N, C, H, W = input_data.shape 
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    
    # [(サンプルにゼロパディング), (チャネルにゼロパディング), (上下にゼロパディング), (左右にゼロパディング)]
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') 
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col