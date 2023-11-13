import numpy as np
from functions import *

# ReLUレイヤ
class ReLU:
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

class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad    = pad
        # 中間データ
        self.x = None
        self.col   = None
        self.col_W = None
    def forward(self,x):
        self.x = x
        FN,C,FH,FW = self.W.shape
        N,C,H,W    = self.x.shape
        out_h = int(1+(H+2*self.pad - FH) / self.stride)
        out_w = int(1+(W+2*self.pad - FW) / self.stride)
        
        # im2col(input_data, filter_h, filter_w, stride=1, pad=0)
        col = im2col(x,FH,FW,self.stride,self.pad)
        col_W = self.W.reshape(FN,-1).T # 1行に展開
        out   = np.dot(col,col_W)
        out   = out.reshape(N,out_h,out_w, -1).transpose(0,3,1,2)
        self.col = col
        self.col_W = col_W
        return out
    def backward(self,dout):
        FN,C,FH,FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1,FN)
        self.db = np.sum(dout,axis=0) # バイアスの勾配は縦方向に集約したデータを流す
        self.dW = np.dot(self.col.T,dout)
        self.dW = self.dW.traspose(1,0).reshape(FN,C,FH,FW)
        
        dcol = np.dot(dout,self.col_W.T)
        dx   = col2im(dcol,self.x.shape,FH,FW,self.stride,self.pad)
        return dx
    
class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad    = pad
        self.x      = None
        self.arg_max = None
    def forward(self,x):
        N, C, H, W = x.shape
        out_h = int(1+(H-self.pool_h)/self.stride)
        out_w = int(1+(W-self.pool_w)/self.stride)
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)
        # 最大値
        arg_max = np.max(col, axis=1)
        self.x  = x
        self.arg_max = arg_max
        # 形状を元に戻す
        out = arg_max.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        return out
    
    def backward(self,dout):
        dout = dout.transpose(0,2,3,1)
        pool_size = self.pool_h*self.pool_w
        dmax      = np.zeros((dout.size,pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx