import numpy as np

class Rosenbrock:
    def __init__(self, x0):
        self.f_name = 'rosenbrock'
        self.x0 = x0
        self.x_opt = np.array([1., 1.])
        self.line_search = True

    @staticmethod
    def f(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    @staticmethod
    def grad(x):
        # Calculate Exact Gradient
        grad = np.zeros(2)
        grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        grad[1] = 200 * (x[1] - x[0] ** 2)
        return grad

    @staticmethod
    def hess(x):
        # Calculate Exact Hessian
        hess = np.zeros((2, 2))
        hess[0, 0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
        hess[0, 1] = -400 * x[0]
        hess[1, 0] = -400 * x[0]
        hess[1, 1] = 200
        return hess


class Quadratic:
    def __init__(self, x0):
        self.f_name = 'quadratic'
        self.x0 = x0
        self.x_opt = np.array([4., 0.])
        self.line_search = True

    @staticmethod
    def f(x):
        return (150 * (x[0] * x[1]) ** 2) + (((0.5 * x[0]) + (2 * x[1]) - 2) ** 2)

    @staticmethod
    def grad(x):
        grad = np.zeros(2)
        grad[0] = 600 * x[0] * x[1] ** 2 + 0.5 * (0.5 * x[0] + 2 * x[1] - 2)
        grad[1] = 300 * x[0] ** 2 * x[1] + 2 * (0.5 * x[0] + 2 * x[1] - 2)
        return grad

    @staticmethod
    def hess(x):
        hessian = np.zeros((2,2))
        hessian[0, 0] = 1200 * x[1] ** 2 + 0.5
        hessian[0, 1] = hessian[1, 0] = 600 * x[0] * x[1] + 2
        hessian[1, 1] = 300 * x[0] ** 2 + 8
        return hessian

functions = {
            'rosenbrock':
                {'func': Rosenbrock,
                'starting_points': [np.array([1.2, 1.2]), np.array([-1.2, 1.]), np.array([0.2, 0.8])]
                },
            'quadratic':
               {'func': Quadratic,
                'starting_points': [np.array([-0.2, 1.2]), np.array([3.8, 0.1]), np.array([1.9, 0.6])]
                },
           }