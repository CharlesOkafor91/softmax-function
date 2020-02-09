import numpy as np
import math

# We will write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    SF = []
    exp_sum = 0
    for t in L:
        t_exp = math.exp(t) 
        exp_sum += t_exp #this is to sum all exponents of the values for our denomenator
    for i in L:
        x = (math.exp(i)/exp_sum) #softmax function of eaxh value
        SF.append(x)
    return SF
