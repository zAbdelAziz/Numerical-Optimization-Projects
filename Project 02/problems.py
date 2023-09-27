import numpy as np
import itertools

class SimplexProblem:
    def __init__(self, A, b, c, x_f=None, x_nf=None):
        self.A = A
        self.b = b
        self.c = c
        # Feasible Starting point
        self.x_f = x_f
        # Infeasible starting point
        self.x_nf = x_nf


class QuadraticProgram:
    def __init__(self, m, predefined=False):
        self.m, self.n = m, 2 * m
        if not predefined:
            # Generate M, y
            self.M, self.y = self.generate_my()
            # Generate A with the size 2^n
            self.A = np.array(list(itertools.product([-1,1], repeat=self.n)), dtype=np.float64)
            # binary = np.unpackbits(np.arange(2**self.n, dtype=np.uint8)[:, np.newaxis], axis=1)[:, -self.n:]
            # self.A = np.where(binary == 0, -1, 1).astype(np.float64)
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
            # self.A = np.array(list(itertools.product([-1,1], repeat=self.n)), dtype=np.float64)
            num_combinations = 2 ** self.n
            binary = ((np.arange(num_combinations)[:, None] & (1 << np.arange(2**self.n)[::-1])) > 0).astype(np.uint8)
            self.A = np.where(binary == 0, -1, 1).astype(np.float64)
            # self.A = np.array(list(itertools.product([-1,1], repeat=self.n)), dtype=np.float64)
            print(self.A.shape)

        # Generate b as zeros with size 2^n
        self.b = -np.zeros(2**self.n, dtype=np.float64)

        # Apply the transformation M^TM to get G
        self.G = np.dot(self.M.T, self.M)
        # Apply the transformation - 2 M^T y to get c
        self.c = -2 * np.dot(self.M.T, self.y)

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


class SP1(SimplexProblem):
    def __init__(self):
        self.A = np.array([[1., 2.], [3., 4.]])
        self.b = np.array([5., 6.])
        self.c = np.array([-7., 8.])

        self.x_f = np.array([-4., 4.5])
        self.x_nf = np.array([0., 0.])


class SP2(SimplexProblem):
    def __init__(self):
        self.A = np.array([[10., 11.], [12., 13.]])
        self.b = np.array([14., 15.])
        self.c = np.array([16., 17.])

        self.x_f = np.array([1.4, 0.])
        self.x_nf = np.array([1., 1.])


class SP3(SimplexProblem):
    def __init__(self):
        self.A = np.array([[15., 0.], [0., 20.]])
        self.b = np.array([-25., 13.])
        self.c = np.array([3., -1.])

        self.x_f = np.array([1.5, 1.])
        self.x_nf = np.array([0., 0.])


class SP4(SimplexProblem):
    def __init__(self):
        self.A = np.array([[15., 16.], [17., 18.]])
        self.b = np.array([-8., -9.])
        self.c = np.array([3., -1.])

        self.x_f = np.array([2., 4.])
        self.x_nf = np.array([0., 0.])


class SP5(SimplexProblem):
    def __init__(self):
        self.A = np.array([[3, 1], [1, 2]])
        self.b = np.array([9, 8])
        self.c = np.array([-1, -1])

        self.x_f = np.array([1., 1.])
        self.x_nf = np.array([0., 0.])


class SP6(SimplexProblem):
    def __init__(self):
        self.A = np.hstack((np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 1, 0, 0, -1, 0, 0, 0, -1]]),
                            np.eye(2)))
        self.b = np.array([2., 2.], dtype=np.float64)
        self.c = np.hstack((np.array([1, 1, 1, -1, 1, 1, 1, 1, 1, 1]),
                            np.zeros(2)))

        self.x_f = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.x_nf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class SP7(SimplexProblem):
    def __init__(self):
        self.A = np.hstack((np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 7, 0, 0, -1]]),
                            np.eye(2)))
        self.b = np.array([3., 4.], dtype=np.float64)
        self.c = np.hstack((np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1]),
                            np.zeros(2)))

        self.x_f = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.x_nf = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


class SP8(SimplexProblem):
    def __init__(self):
        self.A = np.hstack((np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                      [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                      [30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                                      [50, 30, 20, 10, 100, 90, 80, 70, 60, 50]]),
                            np.eye(4)))
        self.b = np.array([100, 200, 300, 400], dtype=np.float64)
        self.c = np.hstack((np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
                            np.zeros(6)))

        self.x_f = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.x_nf = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


class SP9(SimplexProblem):
    def __init__(self):
        self.A = np.hstack((np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, -1, 0, 1, 0, -1, 0, 1, 0]]),
                            np.eye(2)))
        self.b = np.array([100., 30.], dtype=np.float64)
        self.c = np.hstack((np.array([1, -1, -1, -1, 1, 1, 1, -1, 1, -1]),
                            np.zeros(2)))

        self.x_f = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.x_nf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class SP10(SimplexProblem):
    def __init__(self):
        self.A = np.hstack((np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, -1, 0, 1, 0, -1, 0, 1, 0]]),
                            np.eye(2)))
        self.b = np.array([2., 5.], dtype=np.float64)
        self.c = np.hstack((np.array([1, 1, -1, -1, 1, 1, 1, -1, 1, 1]),
                            np.zeros(2)))

        self.x_f = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.x_nf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



simplex_problems = [SP1(), SP2(), SP3(), SP4(), SP5(), SP6(), SP7(), SP8(), SP9(), SP10()]
# print(QuadraticProgram(m=4).A.shape)
# print(QuadraticProgram(m=4).A)
