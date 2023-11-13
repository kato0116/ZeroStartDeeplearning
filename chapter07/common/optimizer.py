import numpy as np

# 確率勾配降下法
class SGD:
    def __init__(self,lr=0.1):
        self.lr = lr # 学習率
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr*grads[key]

# 速度vを新たに考慮
# Momentum　モーメンタム
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.v  = None
        self.lr = lr
        self.momentum = momentum
    def update(self,params,grads):
        # 速度vの初期化
        if self.v is None:
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


# 学習係数の減衰
# AdaGrad　アダ・グラッド　Adaはadaptive(適応的)が語源
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h  = None
    def update(self,params,grads):
        delta = 1e-7
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key]*grads[key]
            params[key] -= self.lr*grads[key]/(np.sqrt(self.h[key])+delta)
            
# AdaGrad＋Momentum
# Adam アダム    
class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.t  = 0
        self.lr = lr
        self.m  = None
        self.v  = None
    def update(self,params,grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.t += 1
        lr_t = self.lr*np.sqrt(1.0-self.beta2**self.t) / (1.0-self.beta1**self.t)
        for key in params.keys():
            self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            # self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            # self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_t*self.m[key]/(np.sqrt(self.v[key])+1e-7)
            
        
        
