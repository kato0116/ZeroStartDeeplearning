import numpy as np
import matplotlib.pyplot as plt

# 勾配の計算
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # 勾配を0で初期化
    for i in range(x.size):
        tmp = x[i]
        # f(x+h)
        x[i] = tmp + h
        fxh1 = f(x)
        # f(x-h)
        x[i] = tmp - h
        fxh2 = f(x)

        # xを元に戻す
        x[i] = tmp
        # 勾配
        grad[i] = (fxh1 - fxh2) / (2 * h)  # 修正: '+' を '-' に変更
    return grad

# lrは学習率, step_numは試行回数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x.copy()

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad  # xの更新
        plt.scatter(x[0],x[1])
    return x

def f(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3, 4])
ans = gradient_descent(f, init_x,lr=0.1,step_num=100)
print(ans)
plt.show()