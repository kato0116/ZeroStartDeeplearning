import numpy as np
import matplotlib.pyplot as plt

# シグモイド関数
def Sigmoid(x):
    return 1/(1+np.exp(-x))

# ステップ関数
def step_function(x):
    return np.array(x>0,dtype=np.int)

x = np.arange(-5,5,0.1)
y = Sigmoid(x)

x1 = np.arange(-5,5,0.1)
y1 = step_function(x1)

plt.plot(x,y)
plt.plot(x1,y1)
plt.ylim(-0.1,1.1)
plt.show()