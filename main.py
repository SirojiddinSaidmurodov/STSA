import time

import numpy as np
from scipy.optimize import linprog

start = time.time()
c = np.genfromtxt("c.csv", delimiter=",", usemask=True)
A_ub = np.genfromtxt("aub.csv", delimiter=",", usemask=True)
b_ub = np.genfromtxt("bub.csv", delimiter=",", usemask=True)
A_eq = np.genfromtxt("aeq.csv", delimiter=",", usemask=True)
b_eq = np.genfromtxt("beq.csv", delimiter=",", usemask=True)
print(linprog(c, A_ub, b_ub, A_eq, b_eq))
stop = time.time()
print("Время :")
print(stop - start)
