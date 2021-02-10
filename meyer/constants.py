""" Constant values used throughout """
import numpy as np

P21 = 2 / 36
V = np.arange(21)
P = np.zeros(21)
P[V < 15] = 2 / 36
P[V >= 15] = 1 / 36
P[0] = 0
# Probability to be lower or equal to. Formulation avoids numerical aberrations.
Q = np.array([2 * x / 36 if x < 15 else (x + 14) / 36 for x in range(21)])
Q[0] = 0
