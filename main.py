import numpy as np

import simplexsolver

if __name__ == '__main__':
    sim = simplexsolver.SimplexSolver(func=np.array([1, 2], dtype=float),
                                      bounds_matrix=np.array([[7, 3, -1, 0, 0, 0],
                                                              [2, 7, 0, -1, 0, 0],
                                                              [6, 4, 0, 0, -1, 0],
                                                              [8, 2, 0, 0, 0, -1]], dtype=float),
                                      bounds_vector=np.array([13, 11, 10, 13], dtype=float))
    print(sim)
    sim.solve(debug=True)
