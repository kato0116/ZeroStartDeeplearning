import numpy as np

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5]) # 重み
    b = -0.7                # バイアス
    tmp = np.sum(x*w) + b
    if tmp>=0:
        return 1
    else:
        return 0

x1 = int(input("x1:"))
x2 = int(input("x2:"))
print(AND(x1,x2))