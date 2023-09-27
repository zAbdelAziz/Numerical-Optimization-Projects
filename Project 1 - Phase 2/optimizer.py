import numpy as np
from scipy.optimize import minimize

from tqdm import tqdm


np.seterr(all='ignore')


class Optimizer:
    def __init__(self, f, exact=True, eps=1e-7, max_iter=1000, h=1e-7):
        """
        Initializes the Optimizer object.

        Parameters:
        - f: An object with the following methods: f(x), x0, x_opt, grad(x), hess(x), line_search(x, p, gradient, alpha=1., c=0.3, rho=0.8).
        - exact: A boolean indicating whether to use exact or approximated solutions for gradient and Hessian calculations. Default is True.
        - max_iter: The maximum number of iterations for the optimization algorithm. Default is 1000.
        - eps: A small value used as a termination criterion for the optimization algorithm. Default is 1e-7.
        - h: A small value used in numerical differentiation for gradient approximation. Default is 1e-7.
        """
        self.exact = exact

        self.f = f.f
        self.x0 = f.x0
        self.x_opt = f.x_opt
        self.grad_exact = f.grad
        self.hess_exact = f.hess
        self.line_search = f.line_search

        self.eps = eps
        self.max_iter = max_iter

        self.hd_eps = h

        self.available_methods = ('newton_method', 'cg_linear', 'cg_nonlinear', 'qn')

    def _line_search(self, x, p, gradient, alpha=1., c=0.3, rho=0.8):
        """
        Performs a backtracking line search to find an appropriate step size.

        Parameters:
        - x: The current point in the optimization algorithm.
        - p: The search direction.
        - gradient: The gradient at point x.
        - alpha: The initial step size. Default is 1.0.
        - c: A constant used to define the sufficient decrease condition. Default is 0.3.
        - rho: The factor by which alpha is reduced in each iteration. Default is 0.8.

        Returns:
        - alpha_i: The step size determined by the line search.
        """
        alpha_i = alpha
        while self.f(x + alpha_i * p) > self.f(x) + c * alpha_i * np.dot(gradient, p):
            alpha_i *= rho
        return alpha_i

    def calc_p_trust_region(self, x, hessian, gradient, radius):
        """
        Calculates the trust region constrained step using a quadratic model.

        Parameters:
        - x: The current point in the optimization algorithm.
        - hessian: The Hessian matrix at point x.
        - gradient: The gradient at point x.
        - radius: The radius of the trust region.

        Returns:
        - p.x: The step direction determined by the trust region.
        """
        bounds = [(-radius, radius)] * len(x)
        p = minimize(lambda p: gradient.dot(p) + 0.5 * p.dot(hessian).dot(p), np.zeros_like(x), bounds=bounds, method='L-BFGS-B')
        return p.x

    def newton_method(self, eigen=True, print_progress=False, **kwargs):
        """
        Performs the Newton's method optimization algorithm.

        Parameters:
        - eigen: A boolean indicating whether to use eigenvalue modification for the Hessian matrix. Default is True.
        - print_progress: A boolean indicating whether to print the progress of the optimization. Default is False.
        - kwargs: Additional keyword arguments to be passed to the line search function.

        Returns:
        - x: The optimized point.
        - f_val: The value of the objective function at the optimized point.
        - num_iter: The number of iterations performed.
        """
        print('\t\t- Newton Method:')
        if eigen and not self.exact:
            print('\t\t-- Hessian Modification [Eigenvalue Modification]')
        x = self.x0
        # for i in tqdm(range(self.max_iter)):
        for i in range(self.max_iter):
            # Compute Gradient and Hessian
            if self.exact:
                # Exact Solution
                gradient = self.grad_exact(x)
                hessian = self.hess_exact(x)
            else:
                # Approximated Solution
                gradient = self._grad(x)
                hessian = self._hess(x)
                # Modified Hessian
                if eigen:
                    # Using Modified Eigen Values
                    hessian += self._mod_eigen(hessian)
                # else:
                #     # Using Modified Cholesky
                #     hessian += self._mod_cholesky(hessian)
            # Compute Newton direction
            p = np.linalg.solve(hessian, -gradient)
            # Backtracking Line Search
            alpha_i = self._line_search(x, p, gradient, **kwargs)
            # Update x
            x += alpha_i * p
            # Print Progress
            if print_progress:
                print(f'\t\t\tIteration {i}:\n\t\t\t\t'
                      f'X = {x.tolist()}\n\t\t\t\t'
                      f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                      f'||Gradient|| = {np.linalg.norm(gradient).tolist()}\n\t\t\t\t'
                      f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                      f'||X optimum - X|| = {np.linalg.norm(self.x_opt - x).tolist()}\n\t\t\t\t'
                      f'alpha = {alpha_i}\n\t\t\t\t')
            # Break if Gradient is less than epsilon
            if np.linalg.norm(gradient) < self.eps:
                break
        # Print Last Iteration
        if not print_progress:
            print(f'\t\t\tIteration {i}:\n\t\t\t\t'
                  f'X = {x.tolist()}\n\t\t\t\t'
                  f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                  f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                  f'X optimum - X = {(self.x_opt - x).tolist()}\n\t\t\t\t'
                  f'alpha = {alpha_i}\n\t\t\t\t')
        return x

    def cg_linear(self, print_progress=False):
        """
        Performs conjugate gradient (linear) method for optimization.

        Args:
        - print_progress: Whether to print progress information at each iteration (default: False).

        Returns:
        - x: The optimized point.
        """
        x = self.x0
        # Calculate Hesian and Gradient [First Iteration]
        if self.exact:
            hessian = self.hess_exact(x)
            gradient = self.grad_exact(x)
        else:
            hessian = self._hess(x)
            gradient = self._grad(x)
        p = -gradient
        for i in range(self.max_iter):
            if self.exact:
                hessian = self.hess_exact(x)
            else:
                hessian = self._hess(x)
            # Calculate Alpha
            alpha = np.dot(gradient.T, gradient) / np.dot(p.T, np.dot(hessian, p))
            # Update x
            x = x + alpha * p
            # Keep the previous gradient
            prev_gradient = gradient
            # Calculate Gradient
            if self.exact:
                gradient = self.grad_exact(x)
            else:
                gradient = self._grad(x)
            # Calculate Beta
            beta = np.dot(gradient.T, gradient) / np.dot(prev_gradient.T, prev_gradient)
            # Update p
            p = -gradient + beta * p
            if print_progress:
                print(f'\t\t\tIteration {i}:\n\t\t\t\t'
                      f'X = {x.tolist()}\n\t\t\t\t'
                      f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                      f'||Gradient|| = {np.linalg.norm(gradient).tolist()}\n\t\t\t\t'
                      f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                      f'beta" = {beta.tolist()}\n\t\t\t\t'
                      f'||X optimum - X|| = {np.linalg.norm(self.x_opt - x).tolist()}\n\t\t\t\t'
                      f'alpha = {alpha.tolist()}')
            # Break If the gradient norm < epsilon
            if np.linalg.norm(gradient) < self.eps:
                break
        if not print_progress:
            print(f'\t\t\tIteration {i}:\n\t\t\t\t'
                  f'X = {x.tolist()}\n\t\t\t\t'
                  f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                  f'||Gradient|| = {np.linalg.norm(gradient).tolist()}\n\t\t\t\t'
                  f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                  f'beta" = {beta.tolist()}\n\t\t\t\t'
                  f'||X optimum - X|| = {np.linalg.norm(self.x_opt - x).tolist()}\n\t\t\t\t'
                  f'alpha = {alpha.tolist()}')
        return x

    def cg_nonlinear(self, method='f_r', print_progress=False, **kwargs):
        """
        Performs conjugate gradient (nonlinear) method for optimization.

        Args:
        - method (str, optional): The method for updating the conjugate direction.
                Can be 'f_r' (Fletcher Reeves) or 'p_r' (Polak Ribierre).
        - print_progress: Whether to print progress information at each iteration (default: False).

        Returns:
        - x: The optimized point.
        """
        if not method in ['f_r', 'p_r']:
            raise AttributeError('Method should be either "p_r" or "f_r"')
        x = self.x0
        # Compute Gradient & Negative Gradient
        gradient = self.grad_exact(x) if self.exact else self._grad(x)
        p = -gradient
        k = 0
        for k in range(self.max_iter):
            # Backtracking Line Search
            alpha = self._line_search(x, p, gradient, **kwargs)
            x = x + alpha * p
            if method == 'f_r':
                # Fletcher Reeves Method
                    # Look Forward
                gradient_new = self.grad_exact(x) if self.exact else self._grad(x)
                beta = np.dot(gradient_new, gradient_new) / np.dot(gradient, gradient)
                p = -gradient_new + beta * p
                if print_progress:
                    print(f'\t\t\tIteration {k}:\n\t\t\t\t'
                          f'X = {x.tolist()}\n\t\t\t\t'
                          f'Current Gradient = {gradient.tolist()}\n\t\t\t\t'
                          f'Next Gradient = {gradient_new.tolist()}\n\t\t\t\t'
                          f'|| Gradient || = {np.linalg.norm(gradient)}\n\t\t\t\t'
                          f'||X optimum - X|| = {np.linalg.norm(self.x_opt - x).tolist()}\n\t\t\t\t'
                          f'beta = {beta.tolist()}\n\t\t\t\t'
                          f'alpha = {alpha}')
                gradient = gradient_new
            elif method == 'p_r':
                # Polak Ribierre Method
                    # Look Backward
                g_prev = gradient
                gradient = self.grad_exact(x) if self.exact else self._grad(x)
                beta = np.dot(gradient, gradient - g_prev) / np.dot(g_prev, g_prev)
                p = -gradient + beta * p
                if print_progress:
                    print(f'\t\t\tIteration {k}:\n\t\t\t\t'
                          f'X = {x.tolist()}\n\t\t\t\t'
                          f'Current Gradient = {gradient.tolist()}\n\t\t\t\t'
                          f'Previous Gradient = {g_prev.tolist()}\n\t\t\t\t'
                          f'|| Gradient || = {np.linalg.norm(gradient)}\n\t\t\t\t'
                          f'||X optimum - X|| = {np.linalg.norm(self.x_opt - x).tolist()}\n\t\t\t\t'
                          f'beta" = {beta.tolist()}\n\t\t\t\t'
                          f'alpha = {alpha}')
            else:
                raise NotImplementedError('How did you escape the matrix neo?')
            if np.linalg.norm(gradient) <= self.eps:
                break
        # Print Progress
        if not print_progress:
            print(f'\t\t\tIteration {k}:\n\t\t\t\t'
              f'X = {x.tolist()}\n\t\t\t\t'
              f'Gradient = {gradient.tolist()}\n\t\t\t\t'
              f'alpha = {alpha}\n\t\t\t\t'
              f'beta = {beta.tolist()}')
        return x

    def qn(self, method='bfgs', trust_region=False, radius=1.0, print_progress=False, **kwargs):
        """
        Performs Quasi-Newton method for optimization.

        Args:
        - method (str, optional): The method for updating the conjugate direction.
                Can be 'bfgs' (Fletcher Reeves) or 'sr1' (Polak Ribierre).
        - trust_region: either use trust region framework or not. [Only works with sr1 - No effect when using bfgs]
        - radius used in trust region framework.
        - print_progress: Whether to print progress information at each iteration (default: False).

        Returns:
        - x: The optimized point.
        """
        x = self.x0
        # Initialize Hessian approximation as identity matrix [Any Positive Definite Can do the trick]
        hessian = np.eye(x.shape[0])
        gradient = self.grad_exact(x) if self.exact else self._grad(x)
        k = 0
        while np.linalg.norm(gradient) > self.eps and k < self.max_iter:
            gradient = self.grad_exact(x) if self.exact else self._grad(x)
            # Compute Search direction
            if trust_region and method == 'sr1':
                p = self.calc_p_trust_region(x, hessian, gradient, radius)
            else:
                p = -np.dot(hessian, gradient)
            # Backtracking Line Search
            alpha = self._line_search(x, p, gradient, **kwargs)
            alpha_i = alpha * p
            x_new = x + alpha_i
            # Calculate Gradient Difference
            if self.exact:
                grad_diff = self.grad_exact(x_new) - gradient
            else:
                grad_diff = self._grad(x_new) - gradient
            if method == 'bfgs':
                # Calculate Rho
                rho = 1 / np.dot(grad_diff, alpha_i)
                # Approximate "Inverse" Hessian [Based on current information of the gradient and previous hessian]
                hessian = self._hess(x, inv=True, rho=rho, alpha_i=alpha_i, grad_diff=grad_diff, hessian=hessian)
                if print_progress:
                    print(f'\t\t\tIteration {k}:\n\t\t\t\t'
                          f'X = {x.tolist()}\n\t\t\t\t'
                          f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                          f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                          f'Gradient Difference = {grad_diff.tolist()}\n\t\t\t\t'
                          f'|| Gradient || = {np.linalg.norm(gradient)}\n\t\t\t\t'
                          f'||X optimum - X|| = {np.linalg.norm(self.x_opt - x).tolist()}\n\t\t\t\t'
                          f'Rho = {rho.tolist()}\n\t\t\t\t'
                          f'alpha = {alpha_i}')
            elif method == 'sr1':
                v = grad_diff - np.dot(hessian, alpha_i)
                va = np.dot(v, alpha_i)
                if np.abs(va) > 1e-8:
                    if not trust_region:
                        hessian += np.outer(v, v) / va
                    else:
                        ha = np.dot(hessian, alpha_i)
                        v_ha = v - ha
                        hessian += np.outer(v_ha, v_ha) / np.dot(v_ha, alpha_i)
                if print_progress:
                    print(f'\t\t\tIteration {k}:\n\t\t\t\t'
                          f'X = {x.tolist()}\n\t\t\t\t'
                          f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                          f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                          f'Gradient Difference = {grad_diff.tolist()}\n\t\t\t\t'
                          f'|| Gradient || = {np.linalg.norm(gradient)}\n\t\t\t\t'
                          f'||X optimum - X|| = {np.linalg.norm(self.x_opt - x).tolist()}\n\t\t\t\t'
                          f'va = {va.tolist()}\n\t\t\t\t'
                          f'alpha = {alpha_i}')
            else:
                raise NotImplementedError('Method Should be either "sr1" or "bfgs" .')
            x = x_new
            k += 1
        if not print_progress:
            if method == 'bfgs':
                print(f'\t\t\tIteration {k}:\n\t\t\t\t'
                      f'X = {x.tolist()}\n\t\t\t\t'
                      f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                      f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                      f'Gradient Difference = {grad_diff.tolist()}\n\t\t\t\t'
                      f'Rho = {rho.tolist()}\n\t\t\t\t'
                      f'alpha = {alpha_i}')
            else:
                print(f'\t\t\tIteration {k}:\n\t\t\t\t'
                      f'X = {x.tolist()}\n\t\t\t\t'
                      f'Gradient = {gradient.tolist()}\n\t\t\t\t'
                      f'Hessian = {hessian.tolist()}\n\t\t\t\t'
                      f'Gradient Difference = {grad_diff.tolist()}\n\t\t\t\t'
                      f'va = {va.tolist()}\n\t\t\t\t'
                      f'alpha = {alpha_i}')
        return x

    def _grad(self, x):
        # Create an array of zeros with the same shape as x to store the gradient
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            # Create a delta array with the same shape as x
            delta = np.zeros_like(x)
            # Set the i-th element of delta to a small value (hd_eps)
            delta[i] = self.hd_eps
            # Calculate the gradient using the central difference formula
            gradient[i] = (self.f(x + delta) - self.f(x - delta)) / (2 * self.hd_eps)
        return gradient

    def _hess(self, x, vector=False, v=None,
              inv=False, rho=None, alpha_i=None, grad_diff=None, hessian=None):
        """
        Calculate the Hessian matrix or perform related operations based on the given parameters.

        Parameters:
            x (array): The point at which to calculate the Hessian or perform related operations.
            vector (bool): Flag indicating whether to calculate the Hessian vector product. Default is False.
            v (array): The vector for the Hessian vector product. Required when vector is True.
            inv (bool): Flag indicating whether to calculate the inverse Hessian. Default is False.
            rho (float): Scalar coefficient for the inverse Hessian calculation. Required when inv is True.
            alpha_i (array): The alpha_i vector for the inverse Hessian calculation. Required when inv is True.
            grad_diff (array): The gradient difference vector for the inverse Hessian calculation. Required when inv is True.
            hessian (array): The initial Hessian matrix. Required when inv is True.

        Returns:
            array: The Hessian matrix or the result of the related operation based on the given parameters.
        """
        n = len(x)
        if not vector:
            # Approximate Hessian
            if not inv:
                hessian = np.zeros((n, n))
                for i in range(n):
                    for j in range(i, n):
                        delta_i = np.zeros_like(x)
                        delta_j = np.zeros_like(x)
                        delta_i[i] = self.hd_eps
                        delta_j[j] = self.hd_eps
                        hessian[i, j] = (self.f(x + delta_i + delta_j)
                                         - self.f(x + delta_i - delta_j)
                                         - self.f(x - delta_i + delta_j)
                                         + self.f(x - delta_i - delta_j)) / (4 * self.hd_eps * self.hd_eps)
                        # Hessian is symmetric
                        hessian[j, i] = hessian[i, j]
            else:
                # Inverse Hessian
                hessian = (np.eye(n) - rho * np.outer(alpha_i, grad_diff)) @ hessian @ (np.eye(n)
                            - rho * np.outer(grad_diff, alpha_i)) + rho * np.outer(alpha_i, alpha_i)

        elif vector:
            # Vector Product Hessian [Remove - using eigen now!]
            hessian = np.zeros_like(x)
            if v is None:
                raise ValueError('Vector should be specified')
            for i in range(n):
                delta = np.zeros_like(x)
                delta[i] = self.hd_eps
                hessian[i] = (self._grad(x + delta) - self._grad(x - delta)).dot(v) / (2 * self.hd_eps)
        else:
            hessian = None
        return hessian

    @staticmethod
    def _mod_cholesky(A):
        # Remove before submission [only 1 midification is required] -> refactor the code
        n = A.shape[0]
        for k in range(n):
            A[k, k] = np.sqrt(max(A[k, k], 0))
            for i in range(k + 1, n):
                A[i, k] = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                for i in range(j, n):
                    A[i, j] = A[i, j] - A[i, k] * A[j, k]
        return A

    @staticmethod
    def _mod_eigen(hessian):
        """
        Modify the eigenvalues of a given Hessian matrix.

        Parameters:
            hessian (array): The Hessian matrix.

        Returns:
            array: The modified Hessian matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eig(hessian)
        # If all eigenvalues are positive, return a matrix of zeros
        if np.all(eigenvalues > 0):
            return np.zeros_like(hessian)
        else:
            # If there are negative eigenvalues, modify the matrix
            min_eigenvalue = np.min(eigenvalues)
            epsilon = 1e-6
            Ek = (abs(min_eigenvalue) + epsilon) * np.eye(hessian.shape[0])
            return Ek