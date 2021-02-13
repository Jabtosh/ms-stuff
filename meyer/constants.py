""" Constant values used throughout """
import numpy as np

L = 21
P21 = 2 / 36
V = np.arange(L)
P = np.zeros(L)
P[V < 15] = 2 / 36
P[V >= 15] = 1 / 36
P[0] = 0
# Probability to be lower or equal to. Formulation avoids numerical aberrations.
Q = np.array([2 * x / 36 if x < 15 else (x + 14) / 36 for x in range(21)])
Q[0] = 0
ppV = {0: "0", 1: "31", 2: "32", 3: "41", 4: "42", 5: "43", 6: "51", 7: "52", 8: "53", 9: "54", 10: "61", 11: "62",
       12: "63", 13: "64", 14: "65", 15: ":11:", 16: ":22:", 17: ":33:", 18: ":44:", 19: ":55:", 20: ":66:"}
