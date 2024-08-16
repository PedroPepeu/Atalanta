import math
from numba import jit

@jit
def relu(x):
    return x if x > 0 else 0

@jit
def sigmoid(x):
    return 1/(1+math.exp(-x))

@jit
def heavyside(x):
    return 0 if x < 0 else 1
