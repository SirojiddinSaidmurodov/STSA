import timeit

import numpy as np
import scipy.optimize as sc

import simplexsolver

A = np.array([1, 2, 0, 0, 0, 0], dtype=float)
B = np.array([[7, 3, -1, 0, 0, 0],
              [2, 7, 0, -1, 0, 0],
              [6, 4, 0, 0, -1, 0],
              [8, 2, 0, 0, 0, -1]], dtype=float)
C = np.array([13, 11, 10, 13], dtype=float)


def simplex():
    simplexsolver.SimplexSolver(func=A,
                                bounds_matrix=B,
                                bounds_vector=C).solve()


def lin_prog():
    sc.linprog(c=A,
               A_eq=B,
               b_eq=C)


if __name__ == '__main__':
    print('Linprog solver runs: ' +
          str(timeit.timeit("lin_prog()", setup="from __main__ import lin_prog", number=1000)) + ' ms')
    print('My simplex solver runs: ' +
          str(timeit.timeit("simplex()", setup="from __main__ import simplex", number=1000)) + ' ms')
