import numpy as np

# ソフトマックス関数
def softmax(x):
    exp_x     = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y         = exp_x/sum_exp_x
    return y

x = np.array([1010,1000,990])
c = max(x)
x = x-c
print(x)
y = softmax(x)
print(y)