""" Constant values used throughout """
import numpy as np


P21 = 2/36
V = np.arange(20)
P = np.zeros(20)
P[V <14] = 2/36
P[V>=14] = 1/36
# to avoid numerical aberrations:
Q = np.array([2*(x+1)/36 if x < 14 else (x+15)/36 for x in range(20)])
