import numpy as np
import scipy.optimize as sc

import simplexsolver

sim = simplexsolver.SimplexSolver(func=np.array([1, 2, 0, 0, 0, 0], dtype=float),
                                  bounds_matrix=np.array([[7, 3, -1, 0, 0, 0],
                                                          [2, 7, 0, -1, 0, 0],
                                                          [6, 4, 0, 0, -1, 0],
                                                          [8, 2, 0, 0, 0, -1]], dtype=float),
                                  bounds_vector=np.array([13, 11, 10, 13], dtype=float))
sim.solve(debug=True)
opt = sc.linprog(c=np.array([1, 2, 0, 0, 0, 0], dtype=float),
                 A_eq=np.array([[7, 3, -1, 0, 0, 0],
                                [2, 7, 0, -1, 0, 0],
                                [6, 4, 0, 0, -1, 0],
                                [8, 2, 0, 0, 0, -1]], dtype=float),
                 b_eq=np.array([13, 11, 10, 13], dtype=float))
print(opt)
