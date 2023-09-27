from datetime import datetime

import numpy as np
from tqdm import tqdm
from scipy import linalg

from problems import *


class ActiveSet:
    def __init__(self, p, max_iter=5000):
        """
        Initialize the ActiveSet solver.

        Args:
            p (QuadraticProgram): Quadratic program instance.
            max_iter (int): Maximum number of iterations for the solver.
        """
        self.G = p.G
        self.c = p.c
        self.A = p.A
        self.b = p.b

        self.max_iter = max_iter

    def run(self, x0):
        """
        Initialize the ActiveSet solver.

        Args:
            p (QuadraticProgram): Quadratic program instance.
            max_iter (int): Maximum number of iterations for the solver.
        """
        x = x0.copy()

        # Initialize Active Constraints
        try:
            # Check if Ax is close to b
            W = np.isclose(np.dot(self.A, x), self.b)
        except:
            # Empty Active Constraints
            W = np.array([], dtype=int)

        for i in range(self.max_iter):
            # Indices of active & inactive constraints
            active_i = np.argwhere(W == True).flatten()
            inactive_i = np.setdiff1d(np.arange(0, W.shape[0]), active_i)

            # Calculate p_k using active constraints [Schur-complement method]
            p_k = self.calc_p(active_i, x)[:len(x)]

            # If p_k is close to zero, check for optimality
            if np.all(np.isclose(p_k, 0)):
                # Calculate gradient g
                g = np.dot(self.G, x) + self.c
                # Solve linear system to get lambda
                l = np.linalg.lstsq(self.A[active_i].T, g, rcond=None)[0]

                # If all lambdas are non-negative, solution found
                if np.all(l >= 0):
                    return i, x
                else:
                    # Otherwise, deactivate constraint with the most negative lambda
                    j = active_i[np.argmin(l)]
                    W[j] = False
            else:
                # Find indices of constraints that violate p_k
                idx = np.argwhere(np.dot(self.A, p_k) < 0).flatten()
                # Keep only the violated constraints among the inactive ones
                idx = np.intersect1d(inactive_i, idx)
                # Keep only the violated constraints among the inactive ones
                _ak = (self.b[idx] - np.dot(self.A[idx], x)) / np.dot(self.A[idx], p_k)
                # Choose the smallest positive ak
                ak = np.min(np.hstack(([1.], _ak)))

                # Update x
                x += ak * p_k
                # If ak is less than 1, activate the corresponding constraint
                if ak < 1:
                    ak_idx = np.argmin(_ak)
                    j = inactive_i[ak_idx]
                    W[j] = True
        return i, x

    def calc_p_kkt(self, active_i, x):
        # KKT method [Something is wrong here!!]
        G = self.G
        A = self.A[active_i]
        c = self.c + np.dot(G, x)
        b = np.zeros(A.shape[0])
        _KKT = np.vstack((np.hstack((G, -A.T)),
                        np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))
                        ))
        kkt = np.hstack((-c, b))
        x_lambda = np.linalg.lstsq(_KKT, kkt, rcond=None)[0]
        # try:
        #     x_lambda = np.linalg.solve(_KKT, kkt)
        # except:
        #     x_lambda = np.linalg.lstsq(_KKT, kkt, rcond=None)[0]
        return x_lambda

    def calc_p(self, active_i, x0, reg_factor=1e-5):
        """
        Calculate the search direction p using the Schur-complement method.

        Args:
            active_i (ndarray): Indices of active constraints.
            x0 (ndarray): Current solution vector.
            reg_factor (float): Regularization factor for the matrix G.

        Returns:
            ndarray: Search direction p.
        """
        # Schur-complement method
        x = x0.copy()
        G = self.G
        c = self.c
        A = self.A[active_i]
        b = np.zeros(len(A))

        # Regularize G to ensure invertibility
        G_reg = G + reg_factor * np.eye(G.shape[0])

        # Regularize G to ensure invertibility
        G_inv = linalg.pinv(G_reg)

        # Calculate g
        g = c + G_reg @ x
        # Calculate h
        h = A @ x - b

        # Solve linear system to get lambda
        lambda_star = np.linalg.lstsq(A @ G_inv @ A.T, A @ G_inv @ g - h, rcond=None)[0]
        # lambda_star = linalg.solve(A @ G_inv @ A.T, A @ G_inv @ g - h)

        # Calculate p_k using lambda
        p_k = linalg.solve(G_reg, A.T @ lambda_star - g)

        return p_k


print('#'*14)
print('# Active Set #')
print('#'*14)
for m in range(1, 6):
    p = QuadraticProgram(m, predefined=False)
    print(f'Problem {m}:')
    print('-'*10)
    print(f'm: {p.m}, n:{p.n}')
    print(f'M:{np.round(p.M, decimals=3).tolist()}')
    print(f'y:{np.round(p.y, decimals=3).tolist()}')
    print(f'A:{p.A.tolist()}')
    print(f'b:{np.round(p.b, decimals=3).tolist()}')
    print(f'c:{np.round(p.c, decimals=3).tolist()}')
    print(f'G:{np.round(p.G, decimals=3).tolist()}')
    print('-'*25)
    for x0 in p.x0s:
        st = datetime.now()
        print(f'X0: {np.round(x0, decimals=3).tolist()}')
        active_set = ActiveSet(p)
        # print('\t',x0)
        i, s = active_set.run(x0)
        print(f'\tIterations: {i+1}')
        print(f'\tSolution: {np.round(s, decimals=3).tolist()}')
        print(f'\tTime Elapsed: {datetime.now() - st}')
        print('\n')
    print('='*20)
    print('\n\n')
