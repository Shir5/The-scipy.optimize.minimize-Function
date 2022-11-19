import numpy as np
from autograd import grad
from math import sqrt
from scipy.optimize import minimize


def func(x):  # Objective function
    return 5 * x[0] ** 2 + 5 * x[1] ** 2 + 6 * x[0] * x[1] - 8 * sqrt(2) * x[0] - 8 * sqrt(2) * x[1]


Df = grad(func)  # Gradient of the objective function
res = minimize(fun=func, x0=np.array([-3., -1., -3., -1.]), jac=Df, method='CG',
               options={'gtol': 10 ** -6, 'disp': True, 'return_all': True})
