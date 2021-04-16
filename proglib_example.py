from scipy.optimize import linprog

obj = [-1, -2]
#      ─┬  ─┬
#       │   └┤ Коэффициент для y
#       └────┤ Коэффициент для x

lhs_ineq = [[2, 1],  # левая сторона красного неравенства
            [-4, 5],  # левая сторона синего неравенства
            [1, -2]]  # левая сторона желтого неравенства

rhs_ineq = [20,  # правая сторона красного неравенства
            10,  # правая сторона синего неравенства
            2]  # правая сторона желтого неравенства

lhs_eq = [[-1, 5]]  # левая сторона зеленого равенства
rhs_eq = [15]
bnd = [(0, float("inf")),  # Границы x
       (0, float("inf"))]  # Границы y
opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
              A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
              method="revised simplex")

print(opt)
