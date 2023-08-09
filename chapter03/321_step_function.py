import numpy as np

def step_function(x):
    if x>0:
        return 1
    else:
        return 0

x = int(input("x:"))
print(step_function(x))