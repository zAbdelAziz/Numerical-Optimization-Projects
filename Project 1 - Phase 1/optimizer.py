import numpy as np
import numdifftools as nd

from tqdm import tqdm


class Optimizer:
    def __init__(self, f, grad=None, hess=None, h0=None):
        """
        Initialize the optimizer with a function f, its gradient grad, and an optional Hessian hess and initial Hessian
        approximation h0 for quasi-Newton methods.
        """
        self.f = f
        if grad is None:
            self.grad = self.gradient
        else:
            self.grad = grad
        if hess is None:
            self.hess = nd.Hessian(f)
        else:
            self.hess = hess
        self.h0 = h0
        self.eps = 1e-6
        self.max_iter = 2000000

    def steepest_descent(self, x0, x_opt=None, alpha=1., rho=0.8, c=0.3, **kwargs):
        """
        Use steepest descent to optimize the function starting from x0, with learning rate alpha, stopping tolerance eps,
        and maximum number of iterations max_iter.
        """
        x = x0
        if x_opt is None:
            raise ValueError('Add x_opt you stupid!!')
        i, grad = 0, 0
        # for i in range(self.max_iter):
        for i in tqdm(range(self.max_iter)):
            grad = self.grad(x, **kwargs)
            alpha_i = alpha
            # print(alpha_i)
            while self.f(x + alpha_i * (-grad), **kwargs) > self.f(x, **kwargs) + c * alpha_i * np.dot(grad, (-grad)):
                alpha_i *= rho
            x -= alpha_i * grad
            if np.any(np.abs(x) > 1e+10):
                raise ValueError('X is very high', x)
            if np.allclose(-grad, 0, self.eps) or np.linalg.norm(grad) < self.eps:
            # if np.linalg.norm(grad) < self.eps and np.allclose(-grad, 0, self.eps):
                break
            if i % 100000 == 0:
                print(x, alpha_i, grad)
        return x, np.abs(x_opt - x), self.f(x, **kwargs), grad, i+1

    def newton(self, x0, x_opt=None, alpha=1., rho=0.8, c=0.3):
        """
        Use Newton's method to optimize the function starting from x0, with stopping tolerance eps and maximum number of
        iterations max_iter.
        """
        x = x0
        if x_opt is None:
            raise ValueError('Add x_opt you stupid!!')
        i, grad = 0, 0
        for i in tqdm(range(self.max_iter)):
        # for i in range(self.max_iter):
            grad = self.grad(x)
            hess = self.hess(x)
            p = - np.linalg.solve(hess, grad)  # Compute Newton direction
            alpha_i = alpha
            while self.f(x + alpha_i * p) > self.f(x) + c * alpha_i * np.dot(grad, p):
                alpha_i *= rho
            # if alpha_i < self.eps:
            #     alpha_i = 1
            x += alpha_i * p

            if np.allclose(-grad, 0, self.eps) or np.linalg.norm(grad) < self.eps:
            # if np.linalg.norm(grad) < self.eps:
                break
            if i % 1000 == 0:
                print(x, alpha_i, grad)
        return x, np.abs(x_opt - x), self.f(x), grad, i+1

    def conjugate_gradient(self, x0):
        """
        Use the conjugate gradient method to optimize the function starting from x0, with stopping tolerance eps and
        maximum number of iterations max_iter.
        """
        x = x0
        r = -self.grad(x)
        p = r
        for i in range(self.max_iter):
            alpha = np.dot(r, r) / np.dot(p, self.hess(x) @ p)
            x += alpha * p
            r_new = r - alpha * (self.hess(x) @ p)
            if np.linalg.norm(r_new) < self.eps:
                break
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new
        return x, self.f(x)

    def quasi_newton(self, x0):
        """
        Use a quasi-Newton method to optimize the function starting from x0, with stopping tolerance eps and maximum
        number of iterations max_iter. The Hessian approximation is updated using the BFGS formula.
        """
        x = x0
        if self.h0 is None:
            h = np.eye(len(x))
        else:
            h = self.h0(x)
        for i in range(self.max_iter):
            grad = self.grad(x)
            if np.linalg.norm(grad) < self.eps:
                break
            d = -np.linalg.solve(h, grad)
            alpha = self._line_search(x, d)
            s = alpha * d
            x_new = x + s
            y = self.grad(x_new) - grad
            if np.abs(np.dot(y, s)) < self.eps:
                break
            rho = 1 / np.dot(y, s)
            h = (np.eye(len(x)) - rho * np.outer(s, y)) @ h @ (np.eye(len(x)) - rho * np.outer(y, s)) + rho * np.outer(s, s)
            x = x_new
        return x, self.f(x)

    def _line_search(self, x, d, alpha_init=1.0, rho=0.5, c=1e-4, max_iter=100):
        """
        Perform a backtracking line search to find a suitable step size for the quasi-Newton method.
        """
        alpha = alpha_init
        for i in range(max_iter):
            if self.f(x + alpha * d) <= self.f(x) + c * alpha * np.dot(self.grad(x), d):
                return alpha
            alpha *= rho
        return alpha

    def gradient(self, x, epsilon=1e-6, **kwargs):
        if not isinstance(x, np.ndarray):
            print('Not a Numpy')
        deriative = np.ndarray(shape=x.shape)
        if x.size != 1:
            for i, xn in enumerate(x):
                epsilon_vector = np.zeros_like(x)
                epsilon_vector[i] = epsilon
                deriative[i] = (self.f(x + epsilon_vector, **kwargs) - self.f(x - epsilon_vector, **kwargs)) / (2 * epsilon)
        else:
            deriative = np.array((self.f(x + epsilon, **kwargs) - self.f(x - epsilon, **kwargs)) / (2 * epsilon))
        return deriative

    def hessian(self, x, vector, epsilon=1e-6):
        if x.size != 1:
            pass
        else:
            hessian_p = (self.derivative(self.f, x, epsilon * vector) - self.derivative(self.f, x)) / epsilon
        return hessian_p

