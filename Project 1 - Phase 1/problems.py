import numpy as np


class PartI:
    def __init__(self):
        self.functions = [self.f1, self.f2, self.f3, self.f4, self.f5]
        self.grads = [self.f1_grad, None, None, None, None]
        self.x0 = [np.array([0.]), np.array([0.]), np.array([0.]), np.array([0.]), np.array([0.])]
        self.x_opt = [-0.589, -1.935, 0.318,0.594,5.5]
        self.alphas = [0.01, 0.01, 0.01,0.001,1.]

    @staticmethod
    def f1(x=np.array([0.])):
        return x ** 5 - 2.5 * x ** 3 + 2 * x + 1

    @staticmethod
    def f1_grad(x):
        return 5 * x ** 4 - 7.5 * x ** 2 + 2

    @staticmethod
    def f2(x=np.array([-1.])):
        return x ** 4 - 3 * x ** 3 - 10 * x ** 2 + 24 * x + 7

    @staticmethod
    def f2_grad(x):
        return 20 * x ** 3 - 15 * x

    @staticmethod
    def f3(x=np.array([1.])):
        return -2 * x * np.exp(-x ** 4) + 5 * np.cos(10 * x)

    @staticmethod
    def f4(x=np.array([-100.])):
        return x ** 3 - 2 * x * np.exp(-x) + np.exp(-2 * x)

    @staticmethod
    def f5(x=np.array([5.])):
        return (x - 3) * (x - 5) * (x - 7) + 85 - (x - 4) * (x - 6) * (x - 8)


class PartII:
    def __init__(self, ns=(1,2,3,4,5), ms=(100, 150, 100, 150, 100), qs=(1,2,1,2,1)):
        self.ns = ns
        self.ms = ms
        self.qs = qs
        self.x0s = [np.zeros(shape=n + 1) for n in self.ns]

        np.random.seed(1234)

    @staticmethod
    def generate_data_points(q, m):
        a = np.random.uniform(-q, q, m)
        b = np.sin(a)
        return a, b

    def sin_nd(self, x, d=2):
        func = 1
        for j in range(d):
            func += np.sin(x[j]) + np.cos(x[j])
        return func

    def least_squares(self, q, m, n):
        ajs, B = self.generate_data_points(q, m)
        def f(x):
            func = 0
            for aj in ajs:
                tmp_func = 0
                for j in range(n + 1):
                    tmp_func += x[j] * aj ** j
                tmp_func -= np.sin(aj)
                tmp_func = tmp_func ** 2
                func += tmp_func
            return func / 2

        def rjs(x):
            rj = []
            for aj in ajs:
                tmp_func = 0
                for j in range(n + 1):
                    tmp_func += x[j] * aj ** j
                tmp_func -= np.sin(aj)
                rj.append(tmp_func)
            return np.asarray(rj)
        return f, rjs, B


class PartIII:
    def __init__(self, n):
        self.n = n
        self.Q = self.hilbert(n)
        self.b = np.ones(n)
        self.x0 = np.zeros(n)

    def hilbert(self, n):
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1 / (i + j + 1)
        return H

    def f(self, x):
        return 0.5 * np.dot(x, np.dot(self.Q, x)) - np.dot(self.b, x)

    def grad_f(self, x):
        return np.dot(self.Q, x) - self.b


class PartIV:
    def __init__(self):
        self.functions = [self.f1, self.f2, self.f3, self.f4, self.f5]
        self.x_opt = np.array([[1., 0.], [0., 0.], [0, 0.], [0., 0.], [0., 0.]])
        self.x_0 = np.array([0.,0.])

    @staticmethod
    def f1(x=np.array([0,0])):
        return 3 * x[0] ** 4 - 4 * x[0] ** 3 + 2 * x[1] ** 2

    @staticmethod
    def f2(x=np.array([0,0])):
        return x[0] ** 4 - 2 * x[0] ** 2 * x[1] + 2 * x[1] ** 2

    @staticmethod
    def f3(x=np.array([0,0])):
        return 3 * x[0] ** 4 - 4 * x[0] ** 3 + 2 * x[1] ** 2 + x[0] ** 2 - x[1] ** 2

    @staticmethod
    def f4(x=np.array([0,0])):
        return x[0] ** 4 + 2 * x[1] ** 4 - 4 * x[0] ** 2 - 4 * x[1] ** 2 + 6
        # return x[0] ** 4 + x[1] ** 4 - 4 * x[0] ** 2 - 4 * x[1] ** 2 + 6

    @staticmethod
    def f5(x=np.array([0,0])):
        return 2 * x[0] ** 4 - 4 * x[0] ** 3 + 2 * x[0] ** 2 + x[1] ** 2
        # return 4 * x[0] ** 4 - 4 * x[0] ** 3 + 4 * x[0] ** 2 * x[1] - 2 * x[1] ** 2
