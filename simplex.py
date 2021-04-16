import logging

import numpy as np


class Simplex:
    __doc__ = "My own realization of simplex algorithm." \
              "Just use solve method"

    def __init__(self,
                 func: np.ndarray,
                 bounds_matrix: np.ndarray,
                 bounds_vector: np.ndarray):
        """
        Initializing object
        :param func : The coefficients of the linear objective function to be minimized.

        :param bounds_matrix : The equality constraint matrix. Each row of ``bounds_matrix`` specifies the
        coefficients of a linear equality constraint on ``x``.

        :param bounds_vector : The equality constraint vector. Each element of ``bounds_matrix @ x`` must equal
        the corresponding element of ``bounds_vector``.
        """
        self.func, self.bounds_matrix, self.bounds_vector = func.copy(), bounds_matrix.copy(), bounds_vector.copy()
        self.simplex_bound_coefficients = None
        self.basis_vars = None
        self.basis: np.ndarray

    def solve(self, debug=False):
        """Solve method
        :param debug True for logging level DEBUG, level INFO otherwise
        """
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.__artificial_vars_add()
        self.__first_phase_solve()

    def __artificial_vars_add(self):
        n, m = self.bounds_matrix.shape
        self.simplex_bound_coefficients = np.hstack(
            (np.column_stack((self.bounds_vector.transpose(), self.bounds_matrix)),
             np.eye(n)))

        self.basis = np.array(range(m + 1, m + n + 1))

        logging.debug('=============================================\n'
                      'artificial variables added:\n' + str(self.simplex_bound_coefficients) + '\nbasis:\n'
                      + str(self.basis) + '\n\n')

    def __first_phase_solve(self):
        n, m = self.bounds_matrix.shape
        self.basis_vars = np.hstack((np.zeros((m + 1,)), np.ones(n, )))
        logging.debug('=============================================\n'
                      'current basis vars:\n' + str(self.basis_vars) + '\n\n')
        for i in range(n):
            rate: np.ndarray = self.__calc_rate()
            rate = rate[1:]
            new_var_column = rate.tolist().index(max(rate))
            new_var_row = self.__find_row_index(p_0=self.simplex_bound_coefficients[:, 0],
                                                p_n=self.simplex_bound_coefficients[:, new_var_column])
            logging.debug('New basis row' + str(new_var_row))

    def __calc_rate(self):
        coefficients = []
        for i in self.basis:
            coefficients.append(self.basis_vars[i])
        coefficients = np.array(coefficients).transpose()
        rate = (np.dot(coefficients, self.simplex_bound_coefficients)) - self.basis_vars
        logging.debug('==============================================\n'
                      'rates:\n' + str(rate) + '\n\n')
        return rate

    def __find_row_index(self, p_0, p_n):
        if (p_n > 0).sum() > 0:
            candidate_indexes: np.ndarray = p_0 / p_n
            can = candidate_indexes.copy()
            can.sort()
            for i in can:
                if i > 0:
                    return candidate_indexes.tolist().index(i)
        else:
            raise Exception('There is no solution')

    def __str__(self):
        return '\n\tfunc:\n' + str(self.func) + \
               '\n\n\tbounds_matrix:\n' + str(self.bounds_matrix) + \
               '\n\n\tbounds_vector:\n' + str(self.bounds_vector)
