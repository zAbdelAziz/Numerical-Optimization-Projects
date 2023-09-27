import numpy as np


class P1:
    def __init__(self, x0):
        self.x0 = x0
        self.c = [self.const_1, self.const_2, self.const_3]
        self.g_c = [self.g_const_1, self.g_const_2, self.g_const_3]
        self.h_c = [self.h_const_1, self.h_const_2, self.h_const_3]
        self.constraints = [{'type': 'eq', 'fun': self.const_1},
                            {'type': 'eq', 'fun': self.const_2},
                            {'type': 'eq', 'fun': self.const_3},]

    def f(self, x):
        xp = np.prod(x)
        print(xp)
        return np.exp(np.prod(x)) - (1/2) * ((x[0]**3 + x[1]**2 + 1) ** 2)

    def g(self, x):
        df_dx0 = np.exp(np.prod(x)) * np.prod(x) - 3 * x[0] * (x[0]**3 + x[1]**2 + 1)
        df_dx1 = np.exp(np.prod(x)) * np.prod(x) - 2 * x[1] * (x[0]**3 + x[1]**2 + 1)
        return np.array([df_dx0, df_dx1, 0, 0, 0])

    def h(self, x):
        # xp = np.prod(x)
        # d2f_dx0x0 = np.exp(xp) * (xp * (xp - 1) - 6 * x[0]**2) - 9 * (x[0]**3 + x[1]**2 + 1)
        # d2f_dx0x1 = np.exp(xp) * (xp * (xp - 1) - 6 * x[0] * x[1])
        # d2f_dx1x1 = np.exp(xp) * (xp * (xp - 1) - 4 * x[1]**2) - 4 * (x[0]**3 + x[1]**2 + 1)
        # return np.array([[d2f_dx0x0, d2f_dx0x1], [d2f_dx0x1, d2f_dx1x1]])
        return hessian

    def const_1(self, x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 - 10

    def const_2(self, x):
        return (x[1] * x[2]) - (5 * x[3] * x[4])

    def const_3(self, x):
        return x[0] ** 3 + x[1] ** 3 + 1

    def g_const_1(self, x):
        return np.array([2*x[0], 2*x[1], 2*x[2], 2*x[3], 2*x[4]])

    def g_const_2(self, x):
        return np.array([0, x[2], x[1], -5*x[4], -5*x[3]])

    def g_const_3(self, x):
        return np.array([3*x[0]**2, 3*x[1]**2, 0, 0, 0])

    def h_const_1(self, x):
        return 2 * np.eye(5)

    def h_const_2(self, x):
        return np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, -5], [0, 0, 0, -5, 0]])

    def h_const_3(self, x):
        return np.array([[6*x[0], 0, 0, 0, 0], [0, 6*x[1], 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])


class P2:
    def __init__(self, x0):
        self.x0 = x0
        self.c = [self.const_1, self.const_2]
        self.g_c = [self.g_const_1, self.g_const_2]
        self.h_c = [self.h_const_1, self.h_const_2]
        self.constraints = [{'type': 'ineq', 'fun': self.const_1},
                            {'type': 'ineq', 'fun': self.const_2},]

    def f(self, x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def g(self, x):
        df_dx0 = 2 * 100 * (x[1] - x[0]**2) * (-2 * x[0]) + 2 * (1 - x[0])
        df_dx1 = 2 * 100 * (x[1] - x[0]**2)
        return np.array([df_dx0, df_dx1])

    def h(self, x):
        # Compute the second-order partial derivatives (elements of the Hessian matrix) of f with respect to x
        d2f_dx0x0 = 2 * (2 * 100 * (x[1] - 3 * x[0]**2) + 2)
        d2f_dx0x1 = -2 * 100 * 2 * x[0]
        d2f_dx1x1 = 2 * 100
        hessian_matrix = np.array([[d2f_dx0x0, d2f_dx0x1],
                                   [d2f_dx0x1, d2f_dx1x1]])

        return hessian_matrix

    def const_1(self, x):
        # c1
        return -x[0]**2 - x[1]**2 + 1

    def const_2(self, x):
        # c2
        return x[1]

    def g_const_1(self, x):
        # Gradient of c1
        np.array([-2 * x[0], -2 * x[1]])

    def g_const_2(self, x):
        # Gradient of c2
        np.array([0, 1])

    def h_const_1(self, x):
        # Hessian of c1
        return np.array([[-2, 0], [0, -2]])

    def h_const_2(self, x):
        # Hessian of c2
        return np.zeros(shape=(2,2))



class P3:
    def __init__(self, x0=None):
        self.x0 = x0
        self.c = [self.const_1, self.const_2, self.const_3]
        self.g_c = [self.g_const_1, self.g_const_2, self.g_const_3]
        self.h_c = [self.h_const_1, self.h_const_2, self.h_const_3]
        self.constraints = [{'type': 'ineq', 'fun': self.const_1},
                            {'type': 'ineq', 'fun': self.const_2},
                            {'type': 'ineq', 'fun': self.const_3},]


    def f(self, x):
        return (150 * (x[0] * x[1]) ** 2) + (0.5 * x[0] + 2 * x[1] - 2) ** 2

    def g(self, x):
        return np.array([300 * x[0] * x[1] * x[1] + 0.25 * x[0] + x[1] - 1, 300 * x[0] * x[0] * x[1] + x[0] + 4 * x[1] - 4])

    def h(self, x):
        return np.array([[300 * (x[1]) ** 2 + 1, 300 * x[0] * x[1]], [300 * (x[1]) ** 2 + 1, 300 * (x[0]) ** 2 + 8]])
        
    def const_1(self, x):
        return (x[0] - 0.5) ** 2 + (x[1] - 1) ** 2 - (5/16)

    def const_2(self, x):
        return (x[0] + 1) ** 2 + (x[1] - (3/8)) ** 2 - (73/64)

    def const_3(self, x):
        return -1 * (x[0] + 1) ** 2 - (x[1] - 1) ** 2 + (np.sqrt(np.array(2)))

    def g_const_1(self, x):
        return np.array([2 * x[0] - 1, 2 * x[1] - 2])

    def g_const_2(self, x):
        return np.array([(2 * x[0]) + 2, (2 * x[1]) - (3/4)])

    def g_const_3(self, x):
        return np.array([(-2 * x[0]) - 2, (-2 * x[1]) + 2])

    def h_const_1(self, x):
        return np.array([[2, 0], [0, 2]])

    def h_const_2(self, x):
        return np.array([[2, 0], [0, 2]])

    def h_const_3(self, x):
        return np.array([[-2, 0], [0, -2]])

class P4:
    def __init__(self, x0):
        self.x0 = x0
        self.c = [self.const_]
        self.constraints = [{'type': 'ineq', 'fun': self.const_},]

    def f(self, x):
        return 0.5 * np.linalg.norm(np.dot(M, x) - y) ** 2

    def const_(self, x):
        return 1 - np.sum(np.abs(x))


class QP:
    def __init__(self, m, predefined=False):
        self.m, self.n = m, 2 * m
        if not predefined:
            # Generate M, y
            self.M, self.y = self.generate_my()
        else:
            self.M = np.array([
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
            ])
            self.y = np.array([1,-2,3,-4,5,-5,4,-3,2,-1]).T
            self.m, self.n = self.M.shape

        self.x0s = []
        np.random.seed(0)
        for i in range(5):
            self.x0s.append(np.abs(np.random.randn(self.n)))
        self.x0s = np.array(self.x0s)

        # print(self.M)

    def generate_my(self):
        """
        Generate random matrices M and vector y.

        Returns:
            ndarray: Matrix M.
            ndarray: Vector y.
        """
        np.random.seed(0)
        x0s = []
        M = np.random.uniform(low=-10, high=10, size=(self.m, self.n))
        y = np.random.uniform(low=-10, high=10, size=(self.m))

        return M, y
