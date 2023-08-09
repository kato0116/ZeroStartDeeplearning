import numpy as np

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp<=0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.3
    tmp = np.sum(x*w) + b
    if tmp<=0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp<=0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y  = AND(s1,s2)
    return y
    
x1 = int(input("x1:"))
x2 = int(input("x2:"))

print("AND:",AND(x1,x2))
print("OR:",OR(x1,x2))
print("NAND:",NAND(x1,x2))
print("XOR",XOR(x1,x2))

    