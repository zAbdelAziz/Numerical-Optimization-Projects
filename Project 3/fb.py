import numpy as np
from datetime import datetime

from problems import *


def soft_threshold(z, lmbda):
    return np.sign(z) * np.maximum(np.abs(z) - lmbda, 0)


def forward_backward_splitting(M, y, x0, tau, alpha, max_iterations=5000, tolerance=1e-8, print_progress=False):
    m, n = M.shape
    x = x0.copy()  # Initialization of x
    for k in range(max_iterations):
        # Forward step
        x_tilde = x - alpha * M.T.dot(M.dot(x) - y)
        # Backward step
        x_next = soft_threshold(x_tilde, tau * alpha)
        if print_progress:
            print(f'\t\t{k+1}: X tilde: {x_tilde.tolist()}, \tX next: {x_next.tolist()}')
        # Check for convergence
        if np.linalg.norm(x_next - x) < tolerance:
            break
        x = x_next
    return k, x

if __name__ == "__main__":
    print('#'*30)
    print('# Forward-Backward Splitting #')
    print('#'*30)
    for m in range(1, 6):
        qp = QP(m=m, predefined=True)
        M, y = qp.M, qp.y
        tau = 1.0
        alpha = 0.001
        print(f'Problem {m}:')
        print('-'*10)
        print(f'm: {qp.m}, n:{qp.n}')
        print(f'M:{np.round(qp.M, decimals=3).tolist()}')
        print(f'y:{np.round(qp.y, decimals=3).tolist()}')
        print(f'alpha: {alpha},\ttau: {tau}')
        print('-'*25)
        for x0 in qp.x0s:
            st = datetime.now()
            print(f'X0: {np.round(x0, decimals=3).tolist()}')
            # print(f'tau: ')
            iteration, result = forward_backward_splitting(M, y, x0, tau, alpha, print_progress=True)
            print(f'\tIterations: {iteration+1}')
            print(f'\tSolution: {np.round(result, decimals=3).tolist()}')
            print(f'\tTime Elapsed: {datetime.now() - st}')
            print('\n')

            # print("Optimal x:", result.tolist(), "at:", iteration)
        print('='*20)
        print('\n\n')
        break
