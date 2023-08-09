# p25
# 2.3.1 簡単な実装

def AND(x1,x2):
    w1, w2, theta = 0.5,0.5,0.7
    tmp = x1*w1 +  x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

x1 = int(input("x1:"))
x2 = int(input("x2:"))

print(AND(x1,x2))
