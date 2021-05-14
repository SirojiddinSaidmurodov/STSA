import logging

import numpy as np


class SimplexSolver:
    __doc__ = "My own realization of two-phase simplex algorithm." \
              "Just use solve method"

    def __init__(self,
                 func: np.ndarray,
                 bounds_matrix: np.ndarray,
                 bounds_vector: np.ndarray, precision=4):
        """
        Initializing the solver
        :param func : The coefficients of the linear objective function to be minimized.

        :param bounds_matrix : The equality constraint matrix. Each row of ``bounds_matrix`` specifies the
        coefficients of a linear equality constraint on ``x``.

        :param bounds_vector : The equality constraint vector. Each element of ``bounds_matrix @ x`` must equal
        the corresponding element of ``bounds_vector``.
        """
        np.set_printoptions(precision=precision, suppress=True, threshold=np.inf, linewidth=np.inf)

        self.func, self.bounds_matrix, self.bounds_vector = func.copy(), bounds_matrix.copy(), bounds_vector.copy()
        self.simplex_bound_coefficients: np.ndarray
        self.obj_function_coefficients: np.ndarray
        self.c_basis: np.ndarray
        self.fun = None
        self.x = None

    def solve(self, debug=False):
        """Solve method
        :param debug True for logging level DEBUG, level INFO otherwise
        """
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.__init_simplex_matrix()
        self.__first_phase_solve()
        self.__second_phase_solve()
        self.__add_results()
        self.__dump()

    def __init_simplex_matrix(self):
        """Inner method for initializing self.simplex_bound_coefficients: adds artificial basis and saves
         their column indexes
        """
        n, m = self.bounds_matrix.shape
        self.simplex_bound_coefficients = np.hstack(
            (np.column_stack((self.bounds_vector.transpose(), self.bounds_matrix)),
             np.eye(n)))
        self.c_basis = np.array(range(m + 1, m + n + 1))

        logging.debug('=============================================\n'
                      'artificial variables added:\n' + str(self.simplex_bound_coefficients)
                      + '\nbasis:\n' + str(self.c_basis) + '\n\n')

    def __str__(self):
        return '\n\tfunc:\n' + str(self.func) + \
               '\n\n\tbounds_matrix:\n' + str(self.bounds_matrix) + \
               '\n\n\tbounds_vector:\n' + str(self.bounds_vector)

    def __first_phase_solve(self):
        n, m = self.bounds_matrix.shape
        self.obj_function_coefficients = np.hstack((np.zeros((m + 1,)), np.ones(n, )))  # m + 1 for calculating delta_0
        logging.debug('=================== FIRST PHASE ==========================\n'
                      'current coefficients of objective function:\n' +
                      str(self.obj_function_coefficients) + '\n\n')
        for i in range(n):
            self.__iterate(m + n)

    def __second_phase_solve(self):
        n, m = self.bounds_matrix.shape
        self.obj_function_coefficients = np.hstack((np.hstack((np.zeros((1,)), self.func)), np.zeros((n,))))
        logging.debug('===================== SECOND PHASE ========================\n'
                      'current coefficients of objective function:\n' +
                      str(self.obj_function_coefficients) + '\n\n')
        logging.debug('=============================================\n'
                      'current simplex bounds:\n' + str(self.simplex_bound_coefficients)
                      + '\nbasis:\n' + str(self.c_basis) + '\n\n')
        while not self.__is_solved(m):
            self.__iterate(m)

    def __iterate(self, end):
        """One iteration of simplex algorithm"""
        rate: np.ndarray = self.__calc_rate()
        rate = rate[1:end]
        new_var_column = rate.tolist().index(max(rate)) + 1
        new_var_row = self.__find_row_index(p_0=self.simplex_bound_coefficients[:, 0],
                                            p_n=self.simplex_bound_coefficients[:, new_var_column])
        logging.debug('New basis row' + str(new_var_row) + '\ncurrent basis' + str(self.c_basis))
        self.simplex_bound_coefficients[new_var_row, :] = \
            self.simplex_bound_coefficients[new_var_row, :] / \
            self.simplex_bound_coefficients[new_var_row, new_var_column]

        self.c_basis[new_var_row] = new_var_column  # writing column index of new basis vector
        self.__calc_c_basis(new_var_row, new_var_column)
        logging.debug('=====================================\n'
                      'current coefficients of objective function:\n' + str(self.obj_function_coefficients) + '\n')
        logging.debug('=====================================\n'
                      'current basis vars:\n' + str(self.simplex_bound_coefficients) + '\n\n')

    @staticmethod
    def __find_row_index(p_0, p_n):
        """Method for finding row index where 1 of the basis vector should appear"""
        if (p_n > 0).sum() > 0:
            candidate_indexes: np.ndarray = p_0 / p_n
            can = candidate_indexes.copy()
            can.sort()
            for i in can:
                if i > 0:
                    return candidate_indexes.tolist().index(i)
        else:
            raise Exception('There is no solution')

    def __calc_rate(self):
        coefficients = []
        for i in self.c_basis:
            coefficients.append(self.obj_function_coefficients[i])
        coefficients = np.array(coefficients).transpose()

        rate = (np.dot(coefficients, self.simplex_bound_coefficients)) - self.obj_function_coefficients
        logging.debug('=====================\n'
                      'rates:\n' + str(rate) + '\n\n')
        return rate

    def __calc_c_basis(self, row, column):
        """Method for recalculating simplex table coefficients"""
        m, n = self.simplex_bound_coefficients.shape
        for i in range(m):
            if i != row:
                self.simplex_bound_coefficients[i, :] = np.subtract(
                    self.simplex_bound_coefficients[i, :],
                    self.simplex_bound_coefficients[i, column].item() *
                    self.simplex_bound_coefficients[row, :])

    def __is_solved(self, end):
        rate: np.ndarray = self.__calc_rate()[1:end] <= 0
        if rate.sum() == rate.size:
            return True
        return False

    def __add_results(self):
        self.fun = self.__calc_rate()[0].item()
        templist = []
        for i in range(1, self.func.size + 1):
            if i in self.c_basis:
                templist.append(self.simplex_bound_coefficients[self.c_basis.tolist().index(i), 0])
            else:
                templist.append(0)
        self.x = np.array(templist)

        logging.debug('\nx: \n' + str(self.x) + '\n\nfunc:' + str(self.fun))

    def analyze(self, precision):
        epsilon = 1 / (10 ** precision)
        results = []
        p_0 = self.simplex_bound_coefficients[:, 0]

        for index in range(p_0.size):
            temp = p_0.copy()
            while True:
                temp[index] -= epsilon
                b_1xp_0 = np.dot(self.optimal_basis, temp)
                if (b_1xp_0 < 0).sum() != 0:
                    break
            infimum = temp[index]
            while True:
                temp[index] += epsilon
                b_1xp_0 = np.dot(self.optimal_basis, temp)
                if (b_1xp_0 < 0).sum() != 0:
                    break
            supremum = temp[index]
            results.append((infimum, supremum))
        return results

    def __dump(self):
        n, m = self.bounds_matrix.shape
        self.optimal_basis = self.simplex_bound_coefficients[:, m + 1:].copy()
        self.optimal_simplex = self.simplex_bound_coefficients.copy()
