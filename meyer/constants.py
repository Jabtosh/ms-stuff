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
THROW_LABEL_MAP = {0: "0", 1: "31", 2: "32", 3: "41", 4: "42", 5: "43", 6: "51", 7: "52", 8: "53", 9: "54", 10: "61",
                   11: "62", 12: "63", 13: "64", 14: "65", 15: ":11:", 16: ":22:", 17: ":33:", 18: ":44:", 19: ":55:",
                   20: ":66:", 21: "M"}
RNG_MAP = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
                    15, 15, 16, 17, 18, 19, 20, 21], dtype=int)
