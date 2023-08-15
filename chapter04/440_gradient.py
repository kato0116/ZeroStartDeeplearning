import numpy as np


def f(x):
    return x[0]**2 + x[1]**2


# 中心差分における微分
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)

# 勾配の計算
def numerical_gradient(f,x):
    h    = 1e-4
    grad = np.zeros_like(x) # 勾配を0で初期化
    for i in range(x.size):
        tmp = x[i]
        # f(x+h)
        x[i] = tmp+h
        fxh1 = f(x)
        # f(x-h)
        x[i] = tmp-h
        fxh2 = f(x)
        
        # xを元に戻す
        x[i] = tmp
        # 勾配
        grad[i] = (fxh1+fxh2)/(2*h)
    return grad


        
        
        