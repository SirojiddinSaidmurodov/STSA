from enum import Enum
from typing import List

import numpy as np


class FunctionType(Enum):
    MIN = 1
    MAX = -1


class Inequality(Enum):
    EQ = '='
    LQ = '<='
    LE = '<'
    GE = '>='
    GR = '>'


class Simplex:

    def __init__(self):
        self._mults_ineq = {Inequality.EQ: 0.0, Inequality.LQ: 1.0, Inequality.LE: 1.0, Inequality.GE: -1.0,
                            Inequality.GR: -1.0}
        self._inv_ineq = {Inequality.EQ: Inequality.EQ,
                          Inequality.LE: Inequality.GR,
                          Inequality.LQ: Inequality.GE,
                          Inequality.GR: Inequality.LE,
                          Inequality.GE: Inequality.LQ}
        self._replacing_index = -1

    def default_inequalities(self, count: int) -> List[Inequality]:
        return [Inequality.EQ for i in range(count)]

    def default_normalized_x(self, count: int) -> List[bool]:
        return [True for i in range(count)]

    def _canonize_B(self):
        for i in range(self.B.size):
            if self.B[i] < 0:
                self.A[i] = -self.A[i]
                self.B[i] = -self.B[i]
                self.inequalities[i] = self._inv_ineq[self.inequalities[i]]

    def _canonize_criterion_function(self):
        if self._f_type == FunctionType.MAX:
            self.C = -self.C

    def _replacing_x(self):
        c_n = 0.0
        A_n = np.zeros(shape=(A.shape[0], 1), dtype=float)
        for i in range(len(self.normalized_x)):
            if not self.normalized_x[i]:
                self._replacing_index = self.C.size + 1

                c_n += self.C[i]
                self.C[i] = -self.C[i]

                A_n += self.A[:, i]
                self.A[:, i] = -self.A[:, i]

        if self._replacing_index != -1:
            self.C = np.hstack((self.C, np.array([c_n], dtype=float)))
            self.A = np.hstack((self.A, A_n))

    def _equalization(self):
        self._x_size = self.C.size
        a_koeffs = []
        for i in range(len(self.inequalities)):
            a_koeffs.append(self._mults_ineq[self.inequalities[i]])

        if a_koeffs:
            self.C = np.hstack((self.C, np.zeros(shape=len(a_koeffs), dtype=float)))
            self.A = np.hstack((self.A, np.diag(a_koeffs)))

    def _artificial_basis(self):
        size = self.B.size
        self.A = np.hstack((self.A, np.eye(size)))
        self.c_i0_index = self.C.size

        self.C_i = np.hstack((np.zeros(self.c_i0_index, dtype=float), np.ones(size, dtype=float)))

    def _canonize(self):
        self._replacing_x()
        self._canonize_criterion_function()
        self._canonize_B()
        self._equalization()

        B0 = np.resize(self.B, new_shape=(self.B.size, 1))
        self.A = np.hstack((B0, self.A))
        self.C = np.hstack((np.array([0.0], dtype=float), self.C))

        self._artificial_basis()

    def _first_positive(self, score_Jordan_Gauss):
        for i in range(1, score_Jordan_Gauss.size):
            if score_Jordan_Gauss[i] > 0:
                return i

        return -1

    def _teta(self, A, JG_index):
        size = A.shape[0] - 1

        P0 = A[:size, 0]
        P_j = A[:size, JG_index]

        min_teta = float('inf')
        min_index = -1

        for i in range(size):
            if P_j[i] > 0.0:
                teta = P0[i] / P_j[i]
                if teta < min_teta:
                    min_teta = teta
                    min_index = i

        return min_index

    def _recalculate_A(self, input_bas_i, output_bas_i):
        self.A[output_bas_i] = self.A[output_bas_i] / self.A[output_bas_i, input_bas_i]
        for i in range(self.A.shape[0]):
            if i == output_bas_i:
                continue
            coeff = -self.A[i, input_bas_i]
            self.A[i] += self.A[output_bas_i] * coeff

    def _recalculates_A(self, border):
        while True:
            input_index = self._first_positive(self.A[-1, :border])
            if input_index == -1:
                break
            output_index = self._teta(self.A, input_index)
            if output_index == -1:
                raise Exception("The system is incompatible!")
            self._recalculate_A(input_index, output_index)
            self.bas_indexes[output_index] = input_index

    def _check_bas_indexes(self):
        for index in self.bas_indexes:
            if index >= self.C.size:
                raise Exception("The system is difficult to solve. "
                                "It is necessary to express an artificial basis as "
                                "a linear combination of non-artificial bases")

    def _calc_i(self):
        self.bas_indexes = [i for i in range(self.c_i0_index, self.C_i.size)]

        C_B = self.C_i[self.bas_indexes]
        score_Jordan_Gauss = C_B.dot(self.A) - self.C_i

        self.A = np.vstack((self.A, np.resize(score_Jordan_Gauss, (1, score_Jordan_Gauss.size))))
        self._recalculates_A(self.C_i.size)

        if self.A[-1, 0] != 0.0:
            raise Exception("The system is incompatible!")
        self._check_bas_indexes()

    def _calc(self):
        self._calc_i()
        self.C = np.hstack((self.C, np.zeros(shape=self.B.size, dtype=float)))

        size = len(self.bas_indexes)
        self.A[size] = self.C[self.bas_indexes].dot(self.A[:size]) - self.C

        self._recalculates_A(self.c_i0_index)

    def _replacing_y(self):
        if self._replacing_index != -1:
            arr = [self.C[self._replacing_index] - self.C[i] for i in range(1, self._replacing_index)]
            self.C = np.hstack((np.array([0.0], dtype=float), np.array([arr], dtype=float)))

    def _recanonize_criterion_function(self):
        if self._f_type == FunctionType.MAX:
            self.C = -self.C
            self.A[-1] = -self.A[-1]

    def _recanonize(self):
        self._recanonize_criterion_function()
        self._replacing_y()

    def solve(self, A: np, B: np, C: np, inequalities: List[Inequality] = None,
              f_type: FunctionType = FunctionType.MIN, normalized_x: List[bool] = None) -> [float, List[float]]:

        self._f_type = f_type
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.inequalities = self.default_inequalities(self.B.size) if inequalities is None else inequalities.copy()
        self.normalized_x = self.default_normalized_x(self.C.size) if normalized_x is None else normalized_x

        self._canonize()
        self._calc()
        self._recanonize()

        X = [0 for i in range(1, self.c_i0_index)]
        for i in range(len(self.bas_indexes)):
            X[self.bas_indexes[i]] = self.A[i][0]

        return self.A[-1, 0], X[1:self._x_size + 1]


A = np.array([[-1, 1, 1, 2, -3],
              [1, 1, 4, 1, -8],
              [0, 1, 1, 0, -4]], dtype=float)
B = np.array([4, 3, -4], dtype=float)
C = np.array([-1, -1, 1, 3, 7], dtype=float)

simplex = Simplex()
s = simplex.solve(A, B, C)
print(s)
